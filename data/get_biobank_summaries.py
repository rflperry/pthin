import gzip
import os
import pandas as pd
import numpy as np

def format_local_gwas(dataset_id, output_csv=None):
    """
    Parses a locally downloaded GWAS VCF file into a clean DataFrame.
    """
    vcf_file = f"{dataset_id}.vcf.gz"
    
    # Failsafe check
    if not os.path.exists(vcf_file):
        raise FileNotFoundError(
            f"File not found! Please manually download {vcf_file} "
            f"from https://opengwas.io/datasets/{dataset_id} and place it in this directory."
        )
                
    print(f"Processing {vcf_file}...")
    data = []
    
    # Iteratively parse the compressed VCF (memory efficient)
    with gzip.open(vcf_file, 'rt') as f:
        for line in f:
            if line.startswith('#'): 
                continue 
            
            parts = line.strip().split('\t')
            if len(parts) < 10: 
                continue
            
            rsid = parts[2]
            fmt_keys = parts[8].split(':')
            fmt_vals = parts[9].split(':')
            record = dict(zip(fmt_keys, fmt_vals))
            
            try:
                es = float(record.get('ES', 0))
                lp = float(record.get('LP', 0))
                
                p_val = 10 ** (-lp)
                effect_sign = np.sign(es)
                
                if p_val <= 1.0 and effect_sign != 0:
                    data.append((rsid, p_val, effect_sign))
                    
            except (ValueError, TypeError):
                continue 

    df = pd.DataFrame(data, columns=['snp', 'p_value', 'effect_sign'])
    
    if output_csv:
        print(f"Saving formatted data to {output_csv}...")
        df.to_csv(output_csv, index=False)
        
    print(f"Finished parsing. Recovered {len(df):,} SNPs.")
    return df

if __name__ == "__main__":
    df = format_local_gwas("ukb-b-19953", "bmi_summary_stats.csv")
    print(df.head())

    df = format_local_gwas("ukb-b-10787", "height_summary_stats.csv")
    print(df.head())

    df = format_local_gwas("ebi-a-GCST006867", "t2d_summary_stats.csv")
    print(df.head())
