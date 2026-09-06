import os
import tarfile
import requests
import pandas as pd
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

ROOT = "BigEarthNet_5000"

METADATA_URL = (
    "https://zenodo.org/records/10891137/files/metadata.parquet"
)

S2_URL = (
    "https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst"
)

METADATA_FILE = os.path.join(
    ROOT,
    "metadata.parquet"
)

S2_ARCHIVE = os.path.join(
    ROOT,
    "BigEarthNet-S2.tar.zst"
)

SUBSET_FILE = os.path.join(
    ROOT,
    "metadata_5000.csv"
)

IMAGE_DIR = os.path.join(
    ROOT,
    "images"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(ROOT, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)


# ============================================================
# DOWNLOAD FUNCTION
# ============================================================

def download_file(url, output_path):

    if os.path.exists(output_path):

        print(
            f"\nAlready exists:\n{output_path}"
        )

        return

    print(
        f"\nDownloading:\n{url}"
    )

    try:

        response = requests.get(
            url,
            stream=True,
            timeout=60
        )

        response.raise_for_status()

    except Exception as e:

        print(
            f"\nDownload failed: {e}"
        )

        raise

    total_size = int(
        response.headers.get(
            "content-length",
            0
        )
    )

    with open(
        output_path,
        "wb"
    ) as file:

        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=os.path.basename(output_path)
        ) as progress:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:

                    file.write(chunk)

                    progress.update(
                        len(chunk)
                    )


# ============================================================
# STEP 1 - DOWNLOAD METADATA
# ============================================================

print("\n" + "=" * 60)
print("[1/4] Downloading metadata")
print("=" * 60)

download_file(
    METADATA_URL,
    METADATA_FILE
)


# ============================================================
# STEP 2 - READ METADATA
# ============================================================

print("\n" + "=" * 60)
print("[2/4] Reading metadata")
print("=" * 60)

df = pd.read_parquet(
    METADATA_FILE
)

print(
    f"Total patches: {len(df):,}"
)

print("\nColumns:")

for column in df.columns:

    print(
        f"  - {column}"
    )


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "patch_id",
    "labels",
    "split",
    "contains_seasonal_snow",
    "contains_cloud_or_shadow"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise RuntimeError(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )


# ============================================================
# STEP 3 - CREATE 5000 PATCH SUBSET
# ============================================================

print("\n" + "=" * 60)
print("[3/4] Selecting 5000 clean patches")
print("=" * 60)


# ------------------------------------------------------------
# Remove snow / cloud / shadow patches
# ------------------------------------------------------------

clean_df = df[
    (df["contains_seasonal_snow"] == False)
    &
    (df["contains_cloud_or_shadow"] == False)
].copy()


print(
    f"\nClean patches available: "
    f"{len(clean_df):,}"
)


# ------------------------------------------------------------
# Normalize split names
# ------------------------------------------------------------

split_values = (
    clean_df["split"]
    .astype(str)
    .str.strip()
    .str.lower()
)


clean_df["_split_normalized"] = split_values


# ------------------------------------------------------------
# Official splits
# ------------------------------------------------------------

train = clean_df[
    clean_df["_split_normalized"] == "train"
].copy()


validation = clean_df[
    clean_df["_split_normalized"].isin(
        [
            "validation",
            "val"
        ]
    )
].copy()


test = clean_df[
    clean_df["_split_normalized"] == "test"
].copy()


print(
    f"Train available: "
    f"{len(train):,}"
)

print(
    f"Validation available: "
    f"{len(validation):,}"
)

print(
    f"Test available: "
    f"{len(test):,}"
)


# ============================================================
# SAMPLE
# ============================================================

TRAIN_COUNT = 3500
VALIDATION_COUNT = 750
TEST_COUNT = 750


if len(train) < TRAIN_COUNT:

    raise RuntimeError(
        f"Not enough training patches. "
        f"Required {TRAIN_COUNT}, "
        f"available {len(train)}"
    )


if len(validation) < VALIDATION_COUNT:

    raise RuntimeError(
        f"Not enough validation patches. "
        f"Required {VALIDATION_COUNT}, "
        f"available {len(validation)}"
    )


if len(test) < TEST_COUNT:

    raise RuntimeError(
        f"Not enough test patches. "
        f"Required {TEST_COUNT}, "
        f"available {len(test)}"
    )


train_small = train.sample(
    n=TRAIN_COUNT,
    random_state=42
)


validation_small = validation.sample(
    n=VALIDATION_COUNT,
    random_state=42
)


test_small = test.sample(
    n=TEST_COUNT,
    random_state=42
)


# ============================================================
# COMBINE
# ============================================================

subset = pd.concat(
    [
        train_small,
        validation_small,
        test_small
    ],
    ignore_index=True
)


# IMPORTANT:
# Do NOT use drop_duplicates() on the entire dataframe.
# labels contains numpy arrays and therefore isn't hashable.
#
# Instead, only deduplicate using patch_id.

subset = subset.drop_duplicates(
    subset=["patch_id"]
)


# Remove helper column

if "_split_normalized" in subset.columns:

    subset = subset.drop(
        columns=["_split_normalized"]
    )


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "-" * 60)

print(
    f"Final selected patches: "
    f"{len(subset):,}"
)

print("-" * 60)


if len(subset) != 5000:

    raise RuntimeError(
        f"Expected exactly 5000 patches, "
        f"but got {len(subset)}"
    )


# ============================================================
# SAVE METADATA
# ============================================================

subset.to_csv(
    SUBSET_FILE,
    index=False
)


print(
    f"\nSaved metadata:\n"
    f"{SUBSET_FILE}"
)


# ============================================================
# PATCH IDs
# ============================================================

patch_ids = set(
    subset["patch_id"]
    .astype(str)
)


print(
    f"\nUnique patch IDs: "
    f"{len(patch_ids):,}"
)


# ============================================================
# STEP 4 - DOWNLOAD S2 ARCHIVE
# ============================================================

print("\n" + "=" * 60)
print("[4/4] BigEarthNet Sentinel-2 archive")
print("=" * 60)

print(
    "\nWARNING:"
)

print(
    "The BigEarthNet-S2 archive is very large "
    "(around 63 GB)."
)

print(
    "Make sure you have sufficient disk space "
    "before continuing."
)

print(
    "\nArchive:"
)

print(
    S2_ARCHIVE
)


answer = input(
    "\nDownload the ~63 GB S2 archive? "
    "[y/N]: "
).strip().lower()


if answer != "y":

    print(
        "\nStopped before downloading S2."
    )

    print(
        "\nYour 5000-patch metadata is ready:"
    )

    print(
        SUBSET_FILE
    )

    print(
        "\nYou can run this script again later "
        "and choose 'y'."
    )

    raise SystemExit


# ============================================================
# DOWNLOAD S2
# ============================================================

download_file(
    S2_URL,
    S2_ARCHIVE
)


# ============================================================
# EXTRACT SELECTED PATCHES
# ============================================================

print("\n" + "=" * 60)
print("Extracting selected patches")
print("=" * 60)


print(
    "\nSelected patch IDs:"
)

print(
    len(patch_ids)
)


# ------------------------------------------------------------
# Match archive members using patch_id
# ------------------------------------------------------------

def is_selected_patch(member_name):

    normalized_name = member_name.replace(
        "\\",
        "/"
    )

    for patch_id in patch_ids:

        if patch_id in normalized_name:

            return True

    return False


# ------------------------------------------------------------
# Stream extraction
# ------------------------------------------------------------

extracted = 0


print(
    "\nScanning archive..."
)

try:

    with tarfile.open(
        S2_ARCHIVE,
        mode="r|zst"
    ) as tar:

        for member in tar:

            if not member.isfile():

                continue


            if not is_selected_patch(
                member.name
            ):

                continue


            tar.extract(
                member,
                path=IMAGE_DIR
            )


            extracted += 1


            if extracted % 100 == 0:

                print(
                    f"Extracted archive entries: "
                    f"{extracted:,}"
                )


except Exception as e:

    print(
        "\nExtraction failed:"
    )

    print(
        e
    )

    raise


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

print(
    f"\nSelected patches: {len(subset):,}"
)

print(
    f"Matching archive entries extracted: "
    f"{extracted:,}"
)

print(
    f"\nDataset directory:"
)

print(
    os.path.abspath(ROOT)
)

print(
    f"\nMetadata:"
)

print(
    os.path.abspath(SUBSET_FILE)
)

print(
    f"\nImages:"
)

print(
    os.path.abspath(IMAGE_DIR)
)

print(
    "\nYour 5000-patch BigEarthNet subset is ready."
)