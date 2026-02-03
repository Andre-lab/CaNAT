genetic_code_RNA2AA = {
    'UUU': 'F', 'UUC': 'F', 'UUA': 'L', 'UUG': 'L',
    'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',
    'AUU': 'I', 'AUC': 'I', 'AUA': 'I', 'AUG': 'M',
    'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',
    'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S',
    'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'UAU': 'Y', 'UAC': 'Y', 'UAA': '*', 'UAG': '*',
    'CAU': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAU': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAU': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'UGU': 'C', 'UGC': 'C', 'UGA': '*', 'UGG': 'W',
    'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGU': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}


def translate_rna_to_amino_acids(rna_sequence):
    """
    Translates an RNA sequence (or DNA sequence) into a string of amino acids based on the genetic code.

    Parameters:
    rna_sequence (str): The RNA sequence (could be DNA if 'T' is replaced with 'U') to be translated.

    Returns:
    str: A string representing the translated amino acid sequence.
         The translation stops when a stop codon ('*') is encountered or the end of the sequence is reached.

    Notes:
    - The function assumes that the input is a valid RNA or DNA sequence (only contains A, U, G, C).
    - If the sequence contains invalid codons, they will be represented as 'X'.
    - If a stop codon ('*') is encountered, translation stops immediately.
    """

    # Replace 'T' with 'U' to handle DNA as well
    rna_sequence = rna_sequence.replace('T', 'U')

    # Initialize an empty string for storing the translated amino acids
    amino_acids = ""

    # Iterate over the RNA sequence in steps of 3 (codon length)
    for i in range(0, len(rna_sequence), 3):
        codon = rna_sequence[i:i + 3]

        # Get the corresponding amino acid from the genetic code
        amino_acid = genetic_code_RNA2AA.get(codon, 'X')  # 'X' represents unknown codons

        # Append the amino acid to the result string
        amino_acids += amino_acid

        # Stop translation if a stop codon ('*') is encountered
        if amino_acid == '*':
            return amino_acids

    return amino_acids


def read_fasta_file(fasta_path):
    """
    Reads a FASTA file and returns sequence IDs and sequences.

    Parameters:
    fasta_path (str): Path to the FASTA file.

    Returns:
    tuple: A tuple containing a list of sequence IDs and a list of sequences.
    """
    ids = []
    sequences = []
    with open(fasta_path, 'r') as file:
        sequence = ""
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if sequence:
                    sequences.append(sequence)
                ids.append(line[1:])
                sequence = ""
            else:
                sequence += line
        if sequence:  # Append the last sequence
            sequences.append(sequence)
    return ids, sequences


def write_fasta_file(ids, sequences, output_file, wrap_length=60):
    """
    Writes sequence IDs and sequences to a FASTA file, wrapping sequences every `wrap_length` characters for readability.

    Parameters:
    ids (list): A list of sequence IDs (headers from the FASTA file).
    sequences (list): A list of sequences to be written.
    output_file (str): Path to the output FASTA file.
    wrap_length (int, optional): The number of characters per line to wrap the sequence. Default is 60.

    Returns:
    None: The function writes the data to a file and does not return any value.

    Raises:
    ValueError: If the lengths of `ids` and `sequences` do not match.
    """
    # Check if the number of ids and sequences match
    if len(ids) != len(sequences):
        raise ValueError("The number of sequence IDs does not match the number of sequences.")

    # Open the output file for writing
    with open(output_file, 'w') as file:
        for seq_id, seq in zip(ids, sequences):
            file.write(f">{seq_id}\n")
            # Wrap the sequence every `wrap_length` characters for better readability
            for i in range(0, len(seq), wrap_length):
                file.write(f"{seq[i:i + wrap_length]}\n")


def align_rna_as_proteins(protein, rna):
    """Align RNA sequence to protein sequence, inserting gaps ('---') in RNA where the protein sequence has gaps."""

    # Ensure the RNA sequence length is a multiple of 3 (valid codon lengths)
    if len(rna) % 3 != 0:
        raise ValueError("RNA sequence length must be a multiple of 3 for codon alignment.")

    rna_align = ""
    index_rna = 0

    # Iterate over the protein sequence
    for aa in protein:
        if aa != "-":  # If the amino acid is not a gap
            rna_align += rna[index_rna:index_rna + 3]  # Add corresponding codon from RNA
            index_rna += 3  # Move 3 nucleotides ahead in RNA sequence
        else:
            rna_align += "---"  # Insert gap in RNA sequence where there is a gap in protein

    return rna_align


def translate_sequence(sequence, genetic_code_RNA2AA = genetic_code_RNA2AA):
    """Translate an RNA or DNA sequence into a protein chain.
    Only the first protein is translated.

    Args:
    sequence (str): The RNA or DNA sequence to be translated.
    genetic_code_RNA2AA (dict): A dictionary mapping RNA codons to amino acids.

    Returns:
    str: The translated protein sequence or an error message if no translation is found.
    """

    # Ensure the sequence is long enough for translation
    if len(sequence) < 3:
        return "No protein found (sequence too short)."

    # Handle DNA by converting T to U (if the sequence is DNA)
    sequence = sequence.replace('T', 'U')

    # Check for the start codon 'AUG'
    if 'AUG' not in sequence:
        return "No start codon found."

    # Find the first occurrence of the start codon
    start_index = sequence.find('AUG')
    sequence = sequence[start_index:]

    # Translate the RNA sequence into a protein sequence
    protein_sequence = ''
    for i in range(0, len(sequence), 3):
        codon = sequence[i:i + 3]

        # Handle codons not found in the genetic code (invalid codon)
        amino_acid = genetic_code_RNA2AA.get(codon, None)

        if amino_acid:
            protein_sequence += amino_acid
        else:
            # If an unknown codon is found, discard the entire protein sequence
            return "No valid translation found due to unknown codon."

        # Stop codon (end of translation)
        if amino_acid == '*':
            break

    # If no valid protein sequence is formed, return a meaningful message
    return protein_sequence if protein_sequence else "No valid translation found."


def align_msa_rna_as_proteins(l_id_protein, l_protein, l_id_rna, l_rna):
    """
    Align RNA sequences based on their protein alignments.

    Args:
    l_id_protein (list): A list of protein sequence IDs.
    l_protein (list): A list of protein sequences.
    l_id_rna (list): A list of RNA sequence IDs (corresponds to l_id_protein).
    l_rna (list): A list of RNA sequences to be aligned.

    Returns:
    list: A list of aligned RNA sequences corresponding to each protein sequence.
    """

    # Ensure input lists are of the same length
    if len(l_id_protein) != len(l_protein) or len(l_id_rna) != len(l_rna):
        raise ValueError("Input lists must have the same length.")

    l_rna_align = []
    for n, species in enumerate(l_id_protein):
        # Check if the species IDs match between protein and RNA lists
        if species != l_id_rna[n]:
            raise ValueError(f"Mismatch between protein and RNA sequence IDs at index {n}.")

        # Align RNA sequence to protein alignment
        rna_align = align_rna_as_proteins(l_protein[n], l_rna[n])
        l_rna_align.append(rna_align)

    return l_rna_align



