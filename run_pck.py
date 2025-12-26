import argparse
import pandas as pd
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Compute PCK metrics per category.")
    parser.add_argument("--model-name", type=str, required=True, 
                        help="Model name used during evaluation (e.g., dinov2_vits14)")
    parser.add_argument("--alpha", type=float, default=0.1, 
                        help="Alpha threshold used during evaluation (e.g., 0.1)")
    parser.add_argument("--output", type=str, default=None, 
                        help="Optional path to save the summary table as CSV")
    
    args = parser.parse_args()

    # Convention: metrics/test_results_{model_name}_alpha{alpha}.csv
    filename = f"test_results_{args.model_name}_{args.alpha}.csv"
    csv_path = os.path.join("metrics", filename)

    if not os.path.exists(csv_path):
        print(f"Error: Could not find results file at: {csv_path}")
        print("Did you run the 'eval' command with these exact parameters?")
        sys.exit(1)

    print(f"Loading results from: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # Filter for Visible Keypoints Only
    if 'is_visible' in df.columns:
        n_total = len(df)
        df = df[df['is_visible'] == 1]
        print(f"Filtered invisible keypoints: {n_total} -> {len(df)}")
    else:
        print("Warning: 'is_visible' column not found. Assuming all rows are visible.")

    # Metric 1: Per-Keypoint PCK (Global & Per Category)
    # Formula: Total Correct Points / Total Visible Points
    global_pck_kps = df['is_correct'].mean()
    cat_pck_kps = df.groupby('category')['is_correct'].mean()
    # ---------------------------------------------------------
    # Metric 2: Per-Image PCK (Global & Per Category)
    # Formula: Average of (Correct / Visible) for each image pair
    # grouped by pair_idx (and category to keep the label)
    img_scores = df.groupby(['pair_idx', 'category'])['is_correct'].mean().reset_index()
    
    global_pck_img = img_scores['is_correct'].mean()
    cat_pck_img = img_scores.groupby('category')['is_correct'].mean()

    # Assemble Final Table
    summary = pd.DataFrame({
        'PCK_Keypoint': cat_pck_kps,
        'PCK_Image': cat_pck_img,
        'Num_Images': img_scores['category'].value_counts()
    })
    
    # Sort by PCK Image score
    summary = summary.sort_values('PCK_Image', ascending=False)
    
    print("\n" + "="*80)
    print(f"RESULTS FOR: {args.model_name} (Alpha={args.alpha})")
    print("="*80)
    print(f"{'CATEGORY':<20} | {'PCK (Img)':<12} | {'PCK (Kps)':<12} | {'# IMAGES':<8}")
    print("-" * 80)
    
    for cat, row in summary.iterrows():
        print(f"{cat:<20} | {row['PCK_Image']:.2%}      | {row['PCK_Keypoint']:.2%}      | {row['Num_Images']:<8}")
        
    print("-" * 80)
    print(f"{'OVERALL (Mean)':<20} | {global_pck_img:.2%}      | {global_pck_kps:.2%}      | {len(img_scores)}")
    print("=" * 80)

    # Save to file
    if args.output:
        summary.to_csv(args.output)
        print(f"\nSummary table saved to: {args.output}")

if __name__ == "__main__":
    main()