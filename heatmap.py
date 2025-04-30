import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the correlation result CSV
df = pd.read_csv("data/correlation_result.csv")

# Reshape into matrix form for heatmap
corr_matrix = df.set_index('feature').T

# Plot and save heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title("Correlation Heatmap of Features vs Dropout")
plt.xlabel("Feature")
plt.ylabel("")

# Save the figure to disk
output_path = "correlation_heatmap_metabase.png"
plt.tight_layout()
plt.savefig(output_path)
plt.close()

output_path
