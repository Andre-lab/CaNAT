
d_three_to_one = {
    'Ala': 'A',   # Alanine
    'Cys': 'C',   # Cysteine
    'Asp': 'D',   # Aspartic Acid
    'Glu': 'E',   # Glutamic Acid
    'Phe': 'F',   # Phenylalanine
    'Gly': 'G',   # Glycine
    'His': 'H',   # Histidine
    'Ile': 'I',   # Isoleucine
    'Lys': 'K',   # Lysine
    'Leu': 'L',   # Leucine
    'Met': 'M',   # Methionine
    'Asn': 'N',   # Asparagine
    'Pro': 'P',   # Proline
    'Gln': 'Q',   # Glutamine
    'Arg': 'R',   # Arginine
    'Ser': 'S',   # Serine
    'Thr': 'T',   # Threonine
    'Val': 'V',   # Valine
    'Trp': 'W',   # Tryptophan
    'Tyr': 'Y',   # Tyrosine
}

dict_mpnn_amino_acid_tokens = {'E': 3, 'M': 10, 'P': 12, 'A': 0, 'F': 4, 'T': 16, 'L': 9, 'V': 17,
                'G': 5, 'K': 8, 'D': 2, 'Y': 19, 'N': 11, 'R': 14, 'Q': 13, 'I': 7,
                'C': 1, 'H': 6, 'W': 18, 'S': 15, 'X': 20}


codon_token_dict = {'CUU': 0, 'UAA': 1, 'ACC': 2, 'UGA': 3,
           'AUC': 4, 'AGG': 5, 'UAG': 6, 'UCU': 7,
           'AGU': 8, 'UGC': 9, 'AAG': 10, 'UUA': 11,
           'ACG': 12, 'GCA': 13, 'CAU': 14, 'CCU': 15,
           'GCU': 16, 'AAU': 17, 'GAA': 18, 'CGA': 19,
           'UCC': 20, 'ACA': 21, 'CCA': 22, 'UUC': 23,
           'GUC': 24, 'AAC': 25, 'GGG': 26, 'AUG': 27,
           'UAC': 28, 'UCG': 29, 'CAC': 30, 'GAC': 31,
           'UAU': 32, 'GAG': 33, 'GGU': 34, 'GUU': 35,
           'CGC': 36, 'GGA': 37, 'UCA': 38, 'CGU': 39,
           '---': 40, 'CUA': 41, 'UGG': 42, 'GGC': 43,
           'CAA': 44, 'AAA': 45, 'GAU': 46, 'CUG': 47,
           'GUA': 48, 'CAG': 49, 'ACU': 50, 'AGA': 51,
           'AUA': 52, 'UGU': 53, 'AGC': 54, 'AUU': 55,
           'CCC': 56, 'UUG': 57, 'GUG': 58, 'CCG': 59,
           'UUU': 60, 'GCG': 61, 'CUC': 62, 'CGG': 63,
           'GCC': 64, 'BBB':65, 'ZZZ':66}


dict_reverse_codon_token={v:k for k,v in codon_token_dict.items()}

codon_pad_token = codon_token_dict['---']
aa_pad_token = dict_mpnn_amino_acid_tokens['X']
mask_token = aa_pad_token


genetic_code_RNA2AA = {'UUU': 'F', 'UUC': 'F', 'UUA': 'L', 'UUG': 'L',
    'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',
    'AUU': 'I', 'AUC': 'I', 'AUA': 'I', 'AUG': 'M',
    'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',
    'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S',
    'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'UAU': 'Y', 'UAC': 'Y', 'UAG': 'X', 'UAA': 'X',
    'CAU': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAU': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAU': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'UGU': 'C', 'UGC': 'C',  'UGG': 'W', 'UGA':'X',
    'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGU': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    '---': '-', 'BBB':'B', "ZZZ":"Z"}
