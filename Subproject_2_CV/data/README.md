The purpose of this file is to explain the structure of the dataset files.
Each file represent one special day from the **IDS-2018** dataset.

The data are saved under a parquet format with snappy compression to save some space.

**Columns**:

- **Entropy_1 to Entropy_10**: The average value of the entropy ofthe payload by slice of 10% (Entropy_1 represents the entropy in the first 10% of the payload)
- **Header_bit_1 to Header_bit_144**: A Binary bit (0 or 1) which represents the bit representation of each bytes of the TCP header
- **PayloadEncoding**: a string value which represent under which encoding the payload was sent (ascii, Windows-1252, None...)
- **PayloadStrings**: decoded payload (if possible)
- **Attack**: if known, describe the attack type (SQL injection, XSS, ...)
- **Label**: Benign or Malicious
