import argparse, os
import pandas as pd

from glob import glob
from random import choices as rchoices

from torch import nn, optim, no_grad as tno_grad, load as tload, unsqueeze as tunsqueeze, save as tsave, device as tdevice, tensor as ttensor, bincount as tbincount
from torch.cuda import device_count, is_available, device as cuda_device
from torch.nn import CrossEntropyLoss, DataParallel
from torch.utils.data import DataLoader, Dataset

from network.scripts.CaNAT.model import NonAutoregressiveTransformer
from network.scripts.CaNAT.dataset import AARNADataset

from utils.scripts.functions import translate_vocabulary_nc, tokenize_amino_acid, transformDNA_fixedsize, translate_codon_to_amino_acids, transform_fixedsize
from utils.scripts.variables import dict_mpnn_amino_acid_tokens, codon_token_dict, genetic_code_RNA2AA, aa_pad_token, mask_token, codon_pad_token


def generator(set, common_length: int, WD = os.getcwd()):
    """
    Generator function that yields datasets created from multiple CSV files.

    Args:
        pand (str): Path to the directory containing the CSV files.
        common_length (int): Fixed length for sequence transformation.

    Yields:
        AARNADataset: A dataset created from the merged DataFrame of CSV files.
    """
    list_df = []
    # Use glob to find files matching the pattern
    c=0
    for pandas_dtf in glob(f'{WD}/database_sequences/data/{set}/pand_*'):
        dftmp = pd.read_csv(pandas_dtf)
        list_df.append(dftmp)
        c+=1

    # Concatenate all the DataFrames in the list
    merged_df = pd.concat(list_df, axis=0, ignore_index=True)

    # Clean up the list to free memory
    del list_df

    # Specify the column names for input and label data
    label_column_name = "DNAseq"

    # Create the Dataset and return it

    aa_rna_dataset = AARNADataset(merged_df, label_column_name, transform=transform_fixedsize,
                                      common_length=common_length)
    del merged_df
    return aa_rna_dataset


def calculate_weights_batch(tgt, vocab_size, ignore_index=None):
    """
    Calculate the weights for each token in the batch for loss calculation.

    Args:
        tgt (torch.Tensor): Target tensor (e.g., labels) with shape (batch_size, sequence_length).
        vocab_size (int): The size of the vocabulary.
        ignore_index (int, optional): Index to ignore when calculating weights (e.g., padding).

    Returns:
        torch.FloatTensor: A tensor of weights for each token class.
    """
    # Calculate the frequency of each token in the batch and add 1 to avoid division by zero.
    weights = tbincount(tgt.view(-1), minlength=vocab_size) + 1  # Ensure all tokens are accounted for

    # Calculate total samples while ignoring specified indices (if any).
    if ignore_index is not None:
        valid_tokens = tgt != ignore_index
        total_samples = valid_tokens.sum().item()
    else:
        total_samples = tgt.numel()

    # Calculate the weight for each class.
    weights = total_samples / (weights.float() * vocab_size)

    return weights.to(tgt.device)


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


def valset(model, device, tgt_vocab_size, validation_set, batch_size_val=64, ignore_index=codon_pad_token):
    """
    Evaluate the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        tgt_vocab_size (int): The size of the target vocabulary.
        validation_set (Dataset): The dataset to use for validation.
        batch_size_val (int, optional): The batch size for validation. Default is 64.
        ignore_index (int, optional): The index to ignore in the loss computation. Default is codon_pad_token.

    Returns:
        float: The average validation loss.
    """
    model.eval()
    criterion = CrossEntropyLoss(ignore_index=ignore_index)

    # Create a DataLoader for the validation set
    data_loader = DataLoader(validation_set, batch_size=batch_size_val, shuffle=False)

    total_loss = 0
    num_batches = 0

    with tno_grad():
        for src, tgt in data_loader:
            src, tgt = src.to(device), tgt.to(device)
            outputs = model(src)

            # Flatten targets and outputs
            tgt = tgt.view(-1)
            outputs = outputs.view(-1, tgt_vocab_size)

            # Compute loss
            loss = criterion(outputs, tgt)
            total_loss += loss.item()
            num_batches += 1

    model.train()
    return total_loss / num_batches if num_batches > 0 else float('inf')


def training_process(nameoutput, parameters=None):
    # Initialize training parametersTo adjust
    l_loss = []
    l_loss_val = []
    synthetic_loss_before = 0
    synthetic_loss_after = 0

    # Initialize model, optimizer, scheduler, and other parameters
    print('Initializing model...')
    model, optimizer, device, batchsize, aa_vocab_size, nc_vocab_size, common_length = hyperparameters()
    print('DEVICE', type(device))
    model.to(device)

    num_epochs=1000
    # Load model parameters if provided
    if parameters:
        model.load_state_dict(tload(parameters, map_location=device))
        model.train()
    else:
        # Synthetic initialization
        criterionbatch = CrossEntropyLoss()
        model.train()  # Set the model to training mode before the synthetic initialization
        print("Starting Synthetic Initialization...")

        for ep_synthetic in range(100):
            rna = rchoices(list(codon_token_dict.keys()), k=3 * common_length)
            y = transformDNA_fixedsize(''.join(rna), common_length)
            x = translate_codon_to_amino_acids(y)
            tgt = translate_vocabulary_nc(y)
            src = tokenize_amino_acid(x)
            src = tunsqueeze(ttensor(src), dim=0).to(device)
            tgt = tunsqueeze(ttensor(tgt), dim=0).to(device)

            optimizer.zero_grad()
            output = model(src)
            loss = criterionbatch(output.view(-1, nc_vocab_size), tgt.view(-1))
            loss.backward()
            optimizer.step()

            if ep_synthetic == 0:  # Record loss before synthetic initialization
                synthetic_loss_before = loss.item()

            print(f"Synthetic Loss (epoch {ep_synthetic}): {loss.item()}")

        # Record the loss after synthetic initialization
        synthetic_loss_after = loss.item()
        print(f"Avg Synthetic Loss After Initialization: {synthetic_loss_after:.4f}")

    ### REAL DATA ###

    # Initialize dataset generators
    print("Loading training datasets...")
    training_dataset = generator("trainset", common_length)
    print("Loading validation datasets...")
    validation_set = generator("validationset", common_length)

    ########
    print("start training")
    # Training loop
    for epoch in range(num_epochs):
        data_loader = DataLoader(training_dataset, batch_size=batchsize, shuffle=True, drop_last=True)
        #data_loader = DataLoader(training_dataset, batch_size=2, shuffle=False, drop_last=True)

        epoch_loss = 0.0
        for batch_idx, batch in enumerate(data_loader):
            src, tgt = batch
            #print("batch_idx", batch_idx)
            weightsbatch = calculate_weights_batch(tgt, nc_vocab_size, ignore_index=codon_pad_token)
            src = src.to(device)
            tgt = tgt.to(device)
            weightsbatch = weightsbatch.to(device)
            criterionbatch = CrossEntropyLoss(weight=weightsbatch, ignore_index=codon_pad_token)
            optimizer.zero_grad()
            output = model(src = src, return_attmap=False)
            output = output.to(device)
            loss = criterionbatch(output.view(-1, nc_vocab_size), tgt.view(-1))
            #print(loss.item())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Average loss for this epoch
        avg_loss = epoch_loss / (batch_idx + 1)
        l_loss.append(avg_loss)
        print("avg_loss", avg_loss)
        # Validation loss
        validation_loss = valset(model, device, nc_vocab_size, validation_set)
        l_loss_val.append(validation_loss)

        # Print the training and validation loss in columns with rounded values
        print(f"Epoch {epoch:03d} | Training Loss: {avg_loss:.4f} | Validation Loss: {validation_loss:.4f}")

        # Save model and losses periodically
        if epoch > 0 and epoch % 1 == 0:  # Save parameters every 10 epochs
            os.makedirs("network/scripts/CaNAT/", exist_ok=True)  # Ensure directory exists
            tsave(model.state_dict(), f"network/scripts/CaNAT/parameters_{nameoutput}.pt")

            with open(f"network/scripts/CaNAT/losses_{nameoutput}.txt", 'w') as f:
                f.write(f"Synthetic Loss Before: {synthetic_loss_before:.4f}\n")
                f.write(f"Synthetic Loss After: {synthetic_loss_after:.4f}\n")
                for i, (train_loss, val_loss) in enumerate(zip(l_loss, l_loss_val)):
                    f.write(f"Epoch {i:03d}; Training Loss: {train_loss:.4f}; Validation Loss: {val_loss:.4f}\n")


def parsearg():
    parser = argparse.ArgumentParser()

    # Define command-line arguments
    parser.add_argument("--parameters", "-p",
                        type=str, required=False, help="parameter_path if training ")
    parser.add_argument("--output", "-o",
                        type=str, required=False, default='CaNAT', help="suffix")
    args = parser.parse_args()

    return args.output, args.parameters


if __name__ == "__main__":
    output, parameters = parsearg()
    training_process(output, parameters)
