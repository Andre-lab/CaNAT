
from random import choice as rchoice, choices as rchoices
from utils.scripts.variables import dict_mpnn_amino_acid_tokens,dict_reverse_codon_token, codon_token_dict, genetic_code_RNA2AA, aa_pad_token, mask_token, codon_pad_token
from torch import Tensor


def translate_vocabulary_nc(codon_seq: list[str]) -> list[int]:
    """
    Encode a codon sequence into a list of tokens.
    Args:
        codon_seq (list of str): List of codons (e.g., ['AUG', 'AAG', ...]).
    Returns:
        list of int: List of tokens corresponding to the codons (e.g., [27, 10, ...]).
    """
    # Ensure the codon exists in the dictionary to avoid KeyError. Otherwise return the pad codon
    return [codon_token_dict.get(cod, codon_pad_token) for cod in codon_seq]



def tokenize_amino_acid(aa_seq: list[str]) -> list[int]:
    """
    Encode a AA sequence into a list of tokens.
    Args:
        aa_seq (list of str): List of codons (e.g., ['A', 'M', ...]).
    Returns:
        list of int: List of tokens corresponding to the AA in the ESM system (e.g., [27, 10, ...]).
    """
    # Ensure the AA exists in the dictionary to avoid KeyError. Otherwise return the pad AA
    return [dict_mpnn_amino_acid_tokens.get(aa, aa_pad_token) for aa in aa_seq]


def transformDNA_fixedsize(seq: str, COMMONLENGTH: int) -> list[str]:
    """
    Adjust a DNA sequence to a fixed size. 
    - Shorter sequences are padded with gaps ("---").
    - Longer sequences are truncated to the specified length.

    :param seq: DNA sequence (string).
    :param COMMONLENGTH: Target length for the sequences.
    :return: List of codons of fixed size.
    """
    # Replace 'T' with 'U' in the sequence to convert it from DNA to RNA
    seq = seq.replace('T', 'U')

    # Transform the sequence into a list of codons
    lcod = []
    for i in range(0, len(seq) - (len(seq) % 3), 3):  # Ensure no out-of-range indexing
        lcod.append(seq[i:i + 3])

    lencod = len(lcod)

    # Homogenize the length of sequences
    if lencod > COMMONLENGTH:
        # Random position to cut the sequence
        pos0 = rchoice(range(0, lencod - COMMONLENGTH))
        return lcod[pos0:pos0 + COMMONLENGTH]

    elif lencod == COMMONLENGTH:
        return lcod

    else:  # lencod < COMMONLENGTH
        # Add gaps to pad the sequence
        numbgap = COMMONLENGTH - lencod
        # Randomly decide how many gaps to place before the sequence
        numbgapbefore = rchoice(range(0, numbgap + 1))
        gapbefore = ["---"] * numbgapbefore
        # The remaining gaps go after the sequence
        numbgapafter = numbgap - numbgapbefore
        gapafter = ["---"] * numbgapafter
        return gapbefore + lcod + gapafter


def translate_codon_to_amino_acids(codon_seq: list[str]) -> list[str]:
    """
    Translate a list of RNA codons into a list of amino acids.

    :param codon_seq: List of RNA codons (list of strings), e.g., ['AUG', 'AAG', ...].
    :return: List of corresponding amino acids (list of strings). If a codon is unknown, 'padding' is used.
    """
    amino_acids = []

    for codon in codon_seq:
        # Use the genetic_code_RNA2AA dictionary to find the amino acid
        amino_acid = genetic_code_RNA2AA.get(codon, aa_pad_token)
        amino_acids.append(amino_acid)

    return amino_acids


def transform_fixedsize(seq: str, COMMONLENGTH: int) -> tuple[list[str], list[int]]:
    """
    Transform a DNA sequence into fixed-size token and amino acid sequences token.

    This function first transforms the DNA sequence into a fixed-size sequence of codons, then translates
    those codons into amino acids, and finally tokenizes both sequences.

    :param seq: DNA/RNA sequence as a string.
    :param COMMONLENGTH: Desired length for the fixed-size sequence.
    :return: A tuple containing:
        - `vocabx`: List of tokenized translated amino acids.
        - `vocaby`: List of tokenized codons.
    """
    # Transform DNA sequence into a fixed-size codon sequence
    y = transformDNA_fixedsize(seq, COMMONLENGTH)

    # Translate codons into amino acids
    x = translate_codon_to_amino_acids(y)

    # Translate codons into vocabulary tokens
    vocaby = translate_vocabulary_nc(y)

    # Tokenize the amino acids
    vocabx = tokenize_amino_acid(x)

    return vocabx, vocaby


def decode_output(output: Tensor) -> list[str]:
    """
    Decode a sequence of token IDs into a list of codons.

    This function converts a tensor of token IDs into their corresponding codons based on a provided
    vocabulary dictionary.

    :param output: Tensor containing token IDs.
    :return: List of decoded codons.
    """
    # Decode the sequence
    decoded_rnaseq = []
    for tok in output:
        codon = dict_reverse_codon_token[tok.item()]
        decoded_rnaseq.append(codon)

    return decoded_rnaseq
