import pyshark
import re
import json
import pandas as pd
from urllib.parse import urlparse
from urllib.parse import parse_qs, unquote
import ipaddress
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns

dff = pd.read_csv("data_for_read.csv")

#dataframe info
dff.info()

dff.describe(include='all')

##### EDA #####

#create bar plot
sns.countplot(x='Label', data=dff)
plt.title("Malicious vs Non-Malicious")
#save barplot
plt.savefig("malicious_vs_non_malicious_bar_graph.png", dpi=300, bbox_inches='tight')
#show the plot
plt.show()

# Set figure size first
plt.figure(figsize=(20, 5))
# Create bar plot
sns.countplot(x='Attack', data=dff)
# Save the plot
plt.savefig("attack_distribution.png", dpi=300, bbox_inches='tight')
# Show the plot
plt.show()

#unstanding the distribution of malicious vs benign data points
print(dff['Label'].value_counts(normalize=True))

#Analysis of correlation between the entropy features

entropy_cols = [f"Entropy_{i}" for i in range(1, 11)]
dff[entropy_cols].describe()
dff[entropy_cols].hist(figsize=(15, 6), bins=30)
plt.suptitle("Distribution of Entropy Columns")
plt.tight_layout()
plt.savefig("entropy_distribution_histogram.png", dpi=300, bbox_inches='tight')
plt.show()

# Correlation heatmap
sns.heatmap(dff[entropy_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Between Entropy Features")
plt.tight_layout()
plt.savefig("entropy_correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()

#Analysis of correlation between the entropy features only for Malicious payloads
malicious_df = dff[dff['Label'] == "Malicious"].copy()
malicious_df.shape

entropy_cols = [f"Entropy_{i}" for i in range(1, 11)]
malicious_df[entropy_cols].describe()
malicious_df[entropy_cols].hist(figsize=(15, 6), bins=30)
plt.suptitle("Distribution of Entropy Columns (Malicious requests alone)")
plt.tight_layout()
plt.savefig("malicious_entropy_distribution_histogram.png", dpi=300, bbox_inches='tight')
plt.show()

# Correlation heatmap
sns.heatmap(malicious_df[entropy_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Between Entropy Features (Malicious requests alone)")
plt.tight_layout()
plt.savefig("malicious_entropy_correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()

#Analysis of Header_Bit columns

tcp_cols = [col for col in dff.columns if col.startswith("Header_bit")]

# Check for mostly 0 or 1 bits
bit_sums = dff[tcp_cols].sum().sort_values(ascending=False)
bit_sums.plot(kind='bar', figsize=(32, 8))
plt.title("Distribution of TCP Header Bits (Sum of 1s per Column)")
plt.xlabel("TCP Header Bits")
plt.ylabel("Sum of 1s")

# Save the plot
plt.tight_layout()
plt.savefig("tcp_header_bits_distribution.png", dpi=300, bbox_inches='tight')
#show the plot
plt.show()

tcp_cols = [col for col in malicious_df.columns if col.startswith("Header_bit")]

# Check for mostly 0 or 1 bits
bit_sums = malicious_df[tcp_cols].sum().sort_values(ascending=False)
bit_sums.plot(kind='bar', figsize=(32, 8))
plt.title("Distribution of TCP Header Bits (Sum of 1s per Column) for Malicious attacks only")
plt.xlabel("TCP Header Bits - Mal")
plt.ylabel("Sum of 1s")

# Save the plot
plt.tight_layout()
plt.savefig("malicious_tcp_header_bits_distribution.png", dpi=300, bbox_inches='tight')
#show the plot
plt.show()

#Text analysis of the payload column
Payload_length = dff['PayloadStrings'].fillna("").apply(len)

sns.histplot(Payload_length, bins=50, kde=True)
plt.title("Payload Length Distribution")
plt.show()

#Word cloud analysis Malicious vs Benign
non_mal = " ".join(dff[dff['Label'] == "Malicious"]['PayloadStrings'].dropna().values)
mal = " ".join(dff[dff['Label'] == "Benign"]['PayloadStrings'].dropna().values)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(WordCloud(width=400, height=300, background_color='white').generate(non_mal))
plt.title("Non-Malicious Payloads")

plt.subplot(1, 2, 2)
plt.imshow(WordCloud(width=400, height=300, background_color='white').generate(mal))
plt.title("Malicious Payloads")

# Adjust layout and save
plt.tight_layout()
plt.savefig("payload_wordclouds.png", dpi=300, bbox_inches='tight')

plt.tight_layout()
plt.show()