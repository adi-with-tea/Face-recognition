"""
Run this script from your face recognition folder to patch baselines.py
"""
import re

with open('baselines.py', 'r') as f:
    content = f.read()

# Replace the load_lfw_pairs function with one that handles the Kaggle CSV format
old_func = None

# Find and replace the load_lfw_pairs function
new_func = '''def load_lfw_pairs(pairs_file, img_dir, mismatch_file=None):
    """Load LFW pairs from Kaggle CSV format (matchpairsDevTest.csv / mismatchpairsDevTest.csv)."""
    pairs = []
    
    # Load positive pairs
    if pairs_file and os.path.exists(pairs_file):
        with open(pairs_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)  # skip header
            for row in reader:
                if len(row) < 3:
                    continue
                name, n1, n2 = row[0].strip(), row[1].strip(), row[2].strip()
                img1 = os.path.join(img_dir, name, f"{name}_{int(n1):04d}.jpg")
                img2 = os.path.join(img_dir, name, f"{name}_{int(n2):04d}.jpg")
                if os.path.exists(img1) and os.path.exists(img2):
                    pairs.append((img1, img2, 1))

    # Load negative pairs
    mismatch = mismatch_file or pairs_file.replace('matchpairs', 'mismatchpairs')
    if os.path.exists(mismatch):
        with open(mismatch, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) < 4:
                    continue
                name1, n1, name2, n2 = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()
                img1 = os.path.join(img_dir, name1, f"{name1}_{int(n1):04d}.jpg")
                img2 = os.path.join(img_dir, name2, f"{name2}_{int(n2):04d}.jpg")
                if os.path.exists(img1) and os.path.exists(img2):
                    pairs.append((img1, img2, 0))

    print(f"  {len(pairs)} pairs loaded ({sum(1 for _,_,l in pairs if l==1)} pos, {sum(1 for _,_,l in pairs if l==0)} neg)")
    return pairs
'''

# Check if csv is imported
if 'import csv' not in content:
    content = 'import csv\n' + content

# Replace existing load_lfw_pairs function
import re
pattern = r'def load_lfw_pairs\(.*?\n(?=def |\nclass |\Z)'
match = re.search(pattern, content, re.DOTALL)
if match:
    content = content[:match.start()] + new_func + '\n' + content[match.end():]
    print("Replaced load_lfw_pairs function")
else:
    # Just prepend the new function before first 'def '
    first_def = content.find('\ndef ')
    content = content[:first_def+1] + new_func + '\n' + content[first_def+1:]
    print("Inserted load_lfw_pairs function")

with open('baselines.py', 'w') as f:
    f.write(content)

print("baselines.py patched successfully!")
