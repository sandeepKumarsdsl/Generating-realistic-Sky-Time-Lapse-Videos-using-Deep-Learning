import pandas as pd
import matplotlib.pyplot as plt

# Load the uploaded CSV file
csv_path = "./clip_similarity_scores.csv"
df = pd.read_csv(csv_path)
'''
# Add dummy FVD and LPIPS scores for demonstration
df["FVD"] = [268 + i % 5 for i in range(len(df))]       # Example variation around 268
df["LPIPS"] = [0.31 - 0.001 * i for i in range(len(df))] # Example decreasing LPIPS
'''
# Save updated CSV with FVD and LPIPS
combined_csv_path = "./clip_fvd_lpips_scores.csv"
df.to_csv(combined_csv_path, index=False)

# === Plot 1: CLIP Similarity Histogram ===
plt.figure(figsize=(8, 5))
plt.hist(df["CLIP Similarity"], bins=20, color='skyblue', edgecolor='black')
plt.title("CLIP Similarity Distribution")
plt.xlabel("CLIP Cosine Similarity")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
clip_hist_path = "./clip_similarity_hist.png"
plt.savefig(clip_hist_path)
'''
# === Plot 2: FVD Bar Chart (top 20) ===
top_fvd = df.sort_values("FVD", ascending=False).head(20)
plt.figure(figsize=(10, 6))
plt.barh(top_fvd["Fake Video"], top_fvd["FVD"], color='tomato')
plt.xlabel("FVD Score")
plt.title("Top 20 FVD Scores (Higher = Worse)")
plt.gca().invert_yaxis()
plt.tight_layout()
fvd_bar_path = "./fvd_top20_bar.png"
plt.savefig(fvd_bar_path)

# === Plot 3: LPIPS Line Plot (all videos) ===
plt.figure(figsize=(10, 5))
plt.plot(df["LPIPS"], marker='o', linestyle='-', color='purple')
plt.title("LPIPS Score per Video")
plt.xlabel("Video Index")
plt.ylabel("LPIPS (Lower is better)")
plt.grid(True)
plt.tight_layout()
lpips_plot_path = "./lpips_line_plot.png"
plt.savefig(lpips_plot_path)
'''
# Return paths to the generated files
combined_csv_path, clip_hist_path #, fvd_bar_path, lpips_plot_path
