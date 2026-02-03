import argparse, os
import pandas as pd

from glob import glob
from random import choices as rchoices

from torch import tensor as ttensor, nn, optim, no_grad as tno_grad, load as tload, unsqueeze as tunsqueeze, save as tsave, device as tdevice, tensor as ttensor, bincount as tbincount, int64 as tint64
from torch.cuda import device_count, is_available, device as cuda_device
from torch.nn import CrossEntropyLoss, DataParallel
from torch.utils.data import TensorDataset, DataLoader

from network.scripts.CaNAT.model import NonAutoregressiveTransformer
from network.scripts.CaNAT.dataset import AARNADataset

from utils.scripts.rnatools import read_fasta_file
from utils.scripts.functions import translate_vocabulary_nc, tokenize_amino_acid, transformDNA_fixedsize, translate_codon_to_amino_acids, transform_fixedsize
from utils.scripts.variables import dict_mpnn_amino_acid_tokens, codon_token_dict, genetic_code_RNA2AA, aa_pad_token, mask_token, codon_pad_token



def pad_and_truncate_list(seq, max_length, pad_value='-'):
    if len(seq) > max_length:
        seq = seq[0:max_length]
    else:
        seq = seq + pad_value * (max_length - len(seq))
    return seq



def inputs_pandas(path, common_length: int, WD = os.getcwd()):
    """
    Parameters:
    - path (str): Relative or absolute path to the directory containing CSV files.
    - common_length (int): The fixed sequence length for padding/truncation.
    - WD (str): Working directory (default is the current working directory).
    #
    Returns:
    - list_id (list): List of sequence IDs.
    - tensor_sequences (torch.Tensor): Tensor of tokenized sequences (dtype=int16).
    """
    df = pd.read_csv(path)
    # Specify the column names for input and label data
    label_column_name = "sequenceAA"
    label_id_name = "id"
    list_id = list(df[label_id_name])
    list_sequenceAA = list(df[label_column_name])
    del df
    #
    # Tokenize and cut the sequence
    list_sequenceAA_batch = []
    for seq in list_sequenceAA:
        seq = pad_and_truncate_list(seq, common_length)
        tokens = [dict_mpnn_amino_acid_tokens.get(i, aa_pad_token) for i in seq]
        list_sequenceAA_batch.append(tokens)

    return list_id, ttensor(list_sequenceAA_batch, dtype=tint64)


def inputs_fasta(fasta_file: str, common_length: int):
    """
    Processes a FASTA file containing protein sequences and prepares the data for deep learning models.

    The function:
    1. Read sequences from a FASTA file.
    2. Tokenizes sequences using a predefined dictionary.
    3. Pads or truncates sequences to a fixed length.
    4. Returns a list of sequence IDs and a PyTorch tensor containing the tokenized sequences.

    Parameters:
    - fasta_file (str): Path to the FASTA file containing protein sequences.
    - common_length (int): The fixed sequence length for padding/truncation.

    Returns:
    - list_id (list): List of sequence IDs.
    - tensor_sequences (torch.Tensor): Tensor of tokenized sequences (dtype=int16).
    """
    # read the FASTA file
    list_id, list_sequenceAA = read_fasta_file(fasta_file)

    list_sequenceAA_batch = []

    # Tokenize and normalize sequences
    for seq in list_sequenceAA:
        # Pad or truncate each sequence to the specified length
        seq = pad_and_truncate_list(seq, common_length)
        # Convert sequence to tokens using the predefined dictionary
        tokens = [dict_mpnn_amino_acid_tokens.get(i, aa_pad_token) for i in seq]  # Default to 0 if not found

        list_sequenceAA_batch.append(tokens)

    # Convert to PyTorch tensor
    tensor_sequences = ttensor(list_sequenceAA_batch, dtype=tint64)
    #
    return list_id, tensor_sequences


def hyperparameters():
    # Vocabulary sizes
    aa_vocab_size = max(dict_mpnn_amino_acid_tokens.values()) + 1  # Vocabulary size for source language
    nc_vocab_size = max(codon_token_dict.values()) + 1

    # Model parameters
    dim_model = 512
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6

    # Input tensors
    batchsize = 64
    COMMONLENGTH = 512

    # Create an instance of the NonAutoregressiveTransformer model
    model = NonAutoregressiveTransformer(aa_vocab_size, nc_vocab_size, dim_model, nhead, num_encoder_layers, num_decoder_layers)

    # Check if multiple GPUs are available and wrap the model in DataParallel
    if device_count() > 1:
        print(f"Using {device_count()} GPUs!")
        model = nn.DataParallel(model)

    # Move the model to the appropriate device (CUDA or CPU)
    device = tdevice('cuda:0' if is_available() else 'cpu')
    print('device = ', device)
    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    return model, optimizer, device, batchsize, aa_vocab_size, nc_vocab_size, COMMONLENGTH


inv_dict_mpnn_amino_acid_tokens={v:k for k,v in dict_mpnn_amino_acid_tokens.items()}

def inverse_translate(tokenAA):
    AA = inv_dict_mpnn_amino_acid_tokens.get(tokenAA.item(), '-')
    return AA

def save_output(src, list_id, outputs, nameoutput):
    # Ensure the output directory exists
    os.makedirs(nameoutput, exist_ok=True)

    for i, tokenAAseq in enumerate(src):
        # Initialize the dictionary to store sequence data
        d = {"amino-acid": [],}

        # Initialize codon probability columns
        for codon in codon_token_dict.keys():
            d[codon] = []


        for j, tokenAA in enumerate(tokenAAseq):
            probsOUTPUT = outputs[i, j, :]
            AA = inverse_translate(tokenAA)
            if AA !='-' and AA !='X':
                # Populate DataFrame dictionary
                d["amino-acid"].append(AA)

                for codon, token in codon_token_dict.items():
                    d[codon].append(probsOUTPUT[token].item())
        #
        # Convert dictionary to DataFrame and save as CSV
        df = pd.DataFrame(d)
        list_id[i]=list_id[i].replace('/', '-')
        df.to_csv(f"{nameoutput}/{list_id[i]}.csv", index=False)



def inference(args):
    """Run inference using a trained model."""

    # Read arguments

    output_name = args.output_dir
    parameters = args.parameters
    path_data = args.input
    os.makedirs(output_name, exist_ok=True)

    # Initialize model and related components
    print("Initializing model...")
    model, optimizer, device, batchsize, aa_vocab_size, nc_vocab_size, common_length = hyperparameters()

    print(f"DEVICE: {device}")
    model.to(device)

    # Load trained model parameters without Parallel (suppress the '.module')
    checkpoint = tload(parameters, map_location=device)

    new_state_dict = {}
    for key, value in checkpoint.items():
        new_key = key.replace('module.', '')  # Retirer le préfixe 'module.'
        new_state_dict[new_key] = value

    # load on new model
    model.load_state_dict(new_state_dict)

    model.eval()

    #sequence inputs
    if path_data.endswith('.csv') :
        list_id, total_dataset_tensor = inputs_pandas(path_data, common_length = common_length)
    elif path_data.endswith('.fasta'):
        list_id, total_dataset_tensor = inputs_fasta(path_data, common_length = common_length)
    else:
        list_id, total_dataset_tensor = inputs_pandas(path_data, common_length=common_length)
        #raise ValueError("Format of input wrong, has to be CSV or FASTA")
    dataset = TensorDataset(total_dataset_tensor)

    # Load dataset
    data_loader = DataLoader(dataset, batch_size=batchsize, shuffle=False)

    # Run inference
    with tno_grad():  # Correction ici
        for src in data_loader:
            src = src[0].to(device)
            outputs = model(src)
            save_output(src, list_id, outputs, output_name)
            list_id = list_id[batchsize:]





def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse command-line arguments for training or inference."
    )

    parser.add_argument(
        "-p", "--parameters",
        type=str,
        required=False,
        help="Path to the parameter file (used for training)."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default="tmp",
        help="Output suffix (default: 'tmp')."
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="path to data for inference. Should be either a .csv file or a .fasta file"
    )
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    inference(args)



""" 
with open("tmp/run_predictionCaNAT_validationset.txt", 'w') as f:
    for pandas in glob("database_sequences/data/validationset/pand_*"):
        species = pandas.replace('pand_','').split('/')[-1]
        f.write(f"python network/scripts/CaNAT/inference.py -p network/scripts/CaNAT/parameters_CaNAT.pt -o analyses/predictions_outputs/CaNAT/prediction_model/validationset/{species} -i database_sequences/data/validationset/pand_{species}\n")



with open("tmp/run_predictionCaNAT_trainset.txt", 'w') as f:
    for pandas in glob("database_sequences/data/trainset/pand_*"):
        species = pandas.replace('pand_','').split('/')[-1]
        f.write(f"python network/scripts/CaNAT/inference.py -p network/scripts/CaNAT/parameters_CaNAT.pt -o analyses/predictions_outputs/CaNAT/prediction_model/trainingset/{species} -i database_sequences/data/trainset/pand_{species}\n")


with open("tmp/run_predictionCaNAT_testset.txt", 'w') as f:
    for pandas in glob("database_sequences/data/testset/pand_*"):
        species = pandas.replace('pand_','').split('/')[-1]
        f.write(f"python network/scripts/CaNAT/inference.py -p network/scripts/CaNAT/parameters_CaNAT.pt -o analyses/predictions_outputs/CaNAT/prediction_model/testset/{species} -i database_sequences/data/testset/pand_{species}\n")



with open("tmp/run_predictionCaNAT_validationset.txt", 'w') as f:
    for pandas in glob("database_sequences/data/validationset/pand_*"):
        species = pandas.replace('pand_','').split('/')[-1]
        f.write(f"python network/scripts/CaNAT/inference.py -p network/scripts/CaNAT/parameters_CaNAT.pt -o analyses/predictions_outputs/CaNAT/prediction_model/validationset/{species} -i database_sequences/data/validationset/pand_{species}\n")


 
with open("tmp/run_predictionCaNATEcoli_validationset.txt", 'w') as f:
    for pandas in glob("database_sequences/data/validationset/pand_*"):
        species = pandas.replace('pand_','').split('/')[-1]
        f.write(f"python network/scripts/CaNAT/inference.py -p network/scripts/CaNAT/parameters_CaNAT_Ecoli.pt -o analyses/predictions_outputs/CaNAT/prediction_model_Ecoli/validationset/{species} -i database_sequences/data/validationset/pand_{species}\n")


with open("tmp/run_predictionCaNAT_testset.txt", 'w') as f:
    for pandas in glob("database_sequences/data/testset/pand_*"):
        species = pandas.replace('pand_','').split('/')[-1]
        f.write(f"python network/scripts/CaNAT/inference.py -p network/scripts/CaNAT/parameters_CaNAT.pt -o analyses/predictions_outputs/CaNAT/prediction_model/testset/{species} -i database_sequences/data/testset/pand_{species}\n")





"""