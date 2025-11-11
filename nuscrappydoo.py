import os
import requests
import time
from pathlib import Path
from urllib.parse import quote
import hashlib
from PIL import Image
from io import BytesIO
import json
import re


class FocusedFruitVeggieBuilder:
    """
    Focused scraper for cauliflower and lemon with strict quality filters
    Ensures the fruit/vegetable is the main subject of the image
    """

    def __init__(self, output_dir="VeggiesFruits_Focused"):
        self.output_dir = Path(output_dir)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        # ONLY cauliflower and lemon
        self.categories = {
            'Verduras': ['lettuce']#
            #'Frutas': ['lemon']
        }

        # STRICT negative keywords - reject anything with these
        self.negative_keywords = [
            # People
            'person', 'people', 'hand', 'hands', 'finger', 'fingers', 'face',
            'woman', 'man', 'child', 'baby', 'human', 'girl', 'boy',

            # Prepared food
            'plate', 'dish', 'bowl', 'cooked', 'cooking', 'recipe', 'meal',
            'dinner', 'lunch', 'breakfast', 'prepared', 'food preparation',
            'salad', 'soup', 'stew', 'roasted', 'baked', 'fried', 'grilled',

            # Drinks
            'juice', 'smoothie', 'drink', 'beverage', 'cocktail', 'glass',
            'lemonade', 'water', 'tea',

            # Settings
            'restaurant', 'kitchen', 'chef', 'food truck', 'market', 'stall',
            'basket', 'bag', 'grocery', 'store', 'shop', 'supermarket',

            # Multiple items/scenes
            'variety', 'assortment', 'collection', 'group', 'many', 'several',
            'display', 'arrangement', 'pile',

            # Processed
            'cut', 'sliced', 'chopped', 'diced', 'minced', 'peeled', 'grated',
            'halved', 'quartered',

            # Backgrounds
            'car', 'truck', 'house', 'building', 'street', 'city', 'garden',
            'farm', 'field', 'tree', 'plant', 'growing'
        ]

        # STRICT positive keywords - prefer images with these
        self.positive_keywords = [
            'isolated', 'studio', 'single', 'one', 'whole',
            'fresh', 'raw', 'natural', 'organic', 'closeup', 'close-up',
            'macro', 'detailed', 'clean'
        ]

        self.downloaded_hashes = set()
        self.stats = {
            'Frutas': {'total': 0, 'success': 0, 'duplicates': 0, 'errors': 0, 'filtered': 0},
            'Verduras': {'total': 0, 'success': 0, 'duplicates': 0, 'errors': 0, 'filtered': 0}
        }

    def setup_directories(self):
        """Create directory structure"""
        print("Creating directory structure...")

        for category in self.categories.keys():
            category_path = self.output_dir / category
            category_path.mkdir(parents=True, exist_ok=True)
            print(f"   Created: {category}/")

        print()

    def get_image_hash(self, img_data):
        """Generate MD5 hash to avoid duplicates"""
        return hashlib.md5(img_data).hexdigest()

    def is_valid_image(self, img_data, min_size=(200, 200), max_size=(4000, 4000)):
        """
        Strict validation for image quality
        Ensures proper size, format, and visual characteristics
        """
        try:
            img = Image.open(BytesIO(img_data))

            # Minimum size check
            if img.size[0] < min_size[0] or img.size[1] < min_size[1]:
                return False, "too_small"

            # Maximum size check
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                return False, "too_large"

            # Aspect ratio check (avoid very stretched images)
            aspect_ratio = max(img.size) / min(img.size)
            if aspect_ratio > 2.5:  # More strict
                return False, "bad_aspect_ratio"

            # Format check
            if img.format not in ['JPEG', 'PNG', 'JPG']:
                return False, "bad_format"

            # Brightness check
            img_gray = img.convert('L')
            brightness = sum(img_gray.getdata()) / len(img_gray.getdata())
            if brightness < 30 or brightness > 230:
                return False, "bad_brightness"

            # Contrast check (ensure variation)
            img_array = list(img_gray.getdata())
            if max(img_array) - min(img_array) < 40:
                return False, "low_contrast"

            # Verify not corrupted
            img.verify()

            return True, "valid"
        except Exception as e:
            return False, f"error_{type(e).__name__}"

    def passes_content_filter(self, tags, title="", description=""):
        """
        STRICT content filter
        Ensures the image is ONLY the fruit/vegetable as main subject
        """
        combined_text = f"{tags} {title} {description}".lower()

        # REJECT if ANY negative keyword found
        for keyword in self.negative_keywords:
            if keyword in combined_text:
                return False, f"rejected_{keyword}"

        # Count positive keywords
        positive_score = sum(1 for keyword in self.positive_keywords if keyword in combined_text)

        # STRICT: Must have at least 1 positive keyword
        if positive_score < 1:
            return False, "no_positive_keywords"

        return True, f"approved_score_{positive_score}"

    def download_from_pixabay(self, query, category, max_images=30, api_key=None):
        """
        Download from Pixabay with STRICT filters for focused images
        """
        if not api_key:
            return 0

        print(f"   Searching for '{query}'...")

        # Enhanced search query for FOCUSED images
        search_query = f"{query} isolated white background single"

        url = "https://pixabay.com/api/"
        params = {
            'key': api_key,
            'q': search_query,
            'image_type': 'photo',
            'category': 'food,nature',
            'per_page': min(max_images * 4, 200),  # Request more to filter heavily
            'safesearch': 'true',
            'orientation': 'all',
            'min_width': 800,  # Higher minimum for quality
            'min_height': 600,
            'editors_choice': 'false',
            'order': 'popular'
        }

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if 'hits' not in data or len(data['hits']) == 0:
                print(f"      No results found")
                return 0

            downloaded = 0
            filtered_out = 0
            category_path = self.output_dir / category

            for idx, hit in enumerate(data['hits']):
                if downloaded >= max_images:
                    break

                try:
                    # Get metadata
                    tags = hit.get('tags', '')
                    title = hit.get('user', '')

                    # STRICT CONTENT FILTER
                    passes, reason = self.passes_content_filter(tags, title)
                    if not passes:
                        filtered_out += 1
                        continue

                    # Get image URL (prefer large)
                    img_url = hit.get('largeImageURL') or hit.get('webformatURL')
                    if not img_url:
                        continue

                    # Download
                    img_response = self.session.get(img_url, timeout=15)
                    img_response.raise_for_status()
                    img_data = img_response.content

                    # Check duplicates
                    img_hash = self.get_image_hash(img_data)
                    if img_hash in self.downloaded_hashes:
                        self.stats[category]['duplicates'] += 1
                        continue

                    # STRICT VALIDATION
                    is_valid, validation_reason = self.is_valid_image(img_data)
                    if not is_valid:
                        self.stats[category]['errors'] += 1
                        continue

                    # Save
                    clean_query = re.sub(r'[^\w\s-]', '', query).replace(' ', '_').lower()
                    filename = f"{clean_query}_{downloaded:04d}.jpg"
                    filepath = category_path / filename

                    # Optimize if too large
                    img = Image.open(BytesIO(img_data))
                    if img.size[0] > 1200 or img.size[1] > 1200:
                        img.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
                        img_buffer = BytesIO()
                        img.save(img_buffer, format='JPEG', quality=92)
                        img_data = img_buffer.getvalue()

                    with open(filepath, 'wb') as f:
                        f.write(img_data)

                    self.downloaded_hashes.add(img_hash)
                    downloaded += 1
                    self.stats[category]['success'] += 1
                    self.stats[category]['total'] += 1

                    # Small pause
                    time.sleep(0.5)

                except Exception as e:
                    self.stats[category]['errors'] += 1
                    continue

            self.stats[category]['filtered'] += filtered_out

            if downloaded > 0:
                print(f"      Downloaded: {downloaded} | Filtered: {filtered_out}")
            else:
                print(f"      No valid images (all filtered)")

            return downloaded

        except Exception as e:
            print(f"      Error: {str(e)[:50]}")
            return 0

    def download_from_unsplash(self, query, category, max_images=30, api_key=None):
        """
        Download from Unsplash with STRICT filters
        """
        if not api_key:
            return 0

        print(f"   Searching '{query}' on Unsplash...")

        url = "https://api.unsplash.com/search/photos"
        headers = {
            'Authorization': f'Client-ID {api_key}'
        }
        params = {
            'query': f'{query} isolated white background',
            'per_page': 30,
            'orientation': 'landscape',
            'content_filter': 'high'
        }

        try:
            response = self.session.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if 'results' not in data or len(data['results']) == 0:
                print(f"      No results found")
                return 0

            downloaded = 0
            filtered_out = 0
            category_path = self.output_dir / category

            for result in data['results']:
                if downloaded >= max_images:
                    break

                try:
                    # Get description
                    description = result.get('description', '') or result.get('alt_description', '')
                    tags = result.get('tags', [])
                    tag_text = ' '.join([tag.get('title', '') for tag in tags]) if isinstance(tags, list) else ''

                    # STRICT FILTER
                    passes, _ = self.passes_content_filter(tag_text, description=description)
                    if not passes:
                        filtered_out += 1
                        continue

                    # Get image URL
                    img_url = result['urls'].get('regular') or result['urls'].get('small')
                    if not img_url:
                        continue

                    # Download
                    img_response = self.session.get(img_url, timeout=15)
                    img_response.raise_for_status()
                    img_data = img_response.content

                    # Check duplicates
                    img_hash = self.get_image_hash(img_data)
                    if img_hash in self.downloaded_hashes:
                        self.stats[category]['duplicates'] += 1
                        continue

                    # Validate
                    is_valid, _ = self.is_valid_image(img_data)
                    if not is_valid:
                        self.stats[category]['errors'] += 1
                        continue

                    # Save
                    clean_query = re.sub(r'[^\w\s-]', '', query).replace(' ', '_').lower()
                    filename = f"{clean_query}_{downloaded:04d}.jpg"
                    filepath = category_path / filename

                    with open(filepath, 'wb') as f:
                        f.write(img_data)

                    self.downloaded_hashes.add(img_hash)
                    downloaded += 1
                    self.stats[category]['success'] += 1
                    self.stats[category]['total'] += 1

                    time.sleep(1)  # Unsplash has stricter limits

                except Exception:
                    self.stats[category]['errors'] += 1
                    continue

            self.stats[category]['filtered'] += filtered_out

            if downloaded > 0:
                print(f"      Downloaded: {downloaded} | Filtered: {filtered_out}")

            return downloaded

        except Exception as e:
            print(f"      Error: {str(e)[:50]}")
            return 0

    def build_focused_dataset(self, images_per_item=150, pixabay_key=None, unsplash_key=None):
        """
        Build dataset with ONLY cauliflower and lemon
        Strict quality control for focused images
        """
        print("=" * 70)
        print("FOCUSED FRUIT/VEGETABLE SCRAPER")
        print("Cauliflower & Lemon Only - High Quality")
        print("=" * 70)
        print()

        if not pixabay_key and not unsplash_key:
            self._print_instructions()
            return

        self.setup_directories()

        print("Configuration:")
        print(f"   Images per type: {images_per_item}")
        print(f"   Pixabay: {'ENABLED' if pixabay_key else 'DISABLED'}")
        print(f"   Unsplash: {'ENABLED' if unsplash_key else 'DISABLED'}")
        print(f"   Quality filters: STRICT (focused images only)")
        print(f"   Items: Cauliflower, Lemon")
        print()

        total_start = time.time()

        for category, items in self.categories.items():
            print(f"\n{'=' * 70}")
            print(f"Category: {category}")
            print(f"{'=' * 70}\n")

            for idx, item in enumerate(items, 1):
                print(f"[{idx}/{len(items)}] {item}")

                downloaded = 0

                # Try Pixabay first
                if pixabay_key and downloaded < images_per_item:
                    downloaded += self.download_from_pixabay(
                        item, category,
                        max_images=images_per_item - downloaded,
                        api_key=pixabay_key
                    )

                # Complement with Unsplash if needed
                if unsplash_key and downloaded < images_per_item:
                    downloaded += self.download_from_unsplash(
                        item, category,
                        max_images=images_per_item - downloaded,
                        api_key=unsplash_key
                    )

                # Pause between searches
                time.sleep(2)

        total_time = time.time() - total_start

        # Summary
        self.print_summary(total_time)

    def _print_instructions(self):
        """Print API key instructions"""
        print("WARNING: YOU NEED AT LEAST ONE API KEY")
        print()
        print("=" * 70)
        print("RECOMMENDED APIs (FREE):")
        print("=" * 70)
        print()
        print("1. PIXABAY (HIGHLY RECOMMENDED)")
        print("   URL: https://pixabay.com/api/docs/")
        print("   Limit: 5,000 requests/hour")
        print("   Registration: 2 minutes")
        print()
        print("2. UNSPLASH (Complementary)")
        print("   URL: https://unsplash.com/developers")
        print("   Limit: 50 requests/hour")
        print()
        print("=" * 70)
        print("USAGE EXAMPLE:")
        print("=" * 70)
        print()
        print("builder = FocusedFruitVeggieBuilder()")
        print("builder.build_focused_dataset(")
        print("    images_per_item=150,")
        print("    pixabay_key='YOUR_PIXABAY_KEY',")
        print("    unsplash_key='YOUR_UNSPLASH_KEY'  # Optional")
        print(")")
        print()
        print("=" * 70)

    def print_summary(self, total_time):
        """Print detailed summary"""
        print("\n" + "=" * 70)
        print("DOWNLOAD SUMMARY")
        print("=" * 70)

        total_success = 0
        total_duplicates = 0
        total_errors = 0
        total_filtered = 0

        for category, stats in self.stats.items():
            if stats['total'] == 0:
                continue

            print(f"\n{category}:")
            print(f"   Success:       {stats['success']:5d}")
            print(f"   Duplicates:    {stats['duplicates']:5d}")
            print(f"   Filtered out:  {stats['filtered']:5d}")
            print(f"   Errors:        {stats['errors']:5d}")
            print(f"   Total proc.:   {stats['total']:5d}")

            total_success += stats['success']
            total_duplicates += stats['duplicates']
            total_errors += stats['errors']
            total_filtered += stats['filtered']

        print(f"\n{'=' * 70}")
        print(f"TOTALS:")
        print(f"   Images saved:       {total_success}")
        print(f"   Filtered (quality): {total_filtered}")
        print(f"   Duplicates avoided: {total_duplicates}")
        print(f"   Errors:             {total_errors}")
        print(f"   Time:               {total_time / 60:.1f} minutes")
        print(f"{'=' * 70}")

        if total_success > 0:
            print(f"\nDataset created in: {self.output_dir}/")
            print(f"\nStructure:")
            print(f"   {self.output_dir}/")
            for category in ['Frutas', 'Verduras']:
                count = self.stats[category]['success']
                if count > 0:
                    print(f"   |-- {category}/")
                    print(f"   |   |-- {count} focused images")
            print()
            print("NEXT STEPS:")
            print("   1. Manually review images")
            print("   2. Remove any incorrect images")
            print("   3. Train: python ImprovedFruitTrainer.py")
        else:
            print("\nNo valid images downloaded.")
            print("   Check your API key")
            print("   Check your connection")
            print("   Filters may be too strict")


def main():
    print("\n" + "=" * 70)
    print("FOCUSED SCRAPER: CAULIFLOWER & LEMON")
    print("=" * 70)

    # Create builder
    builder = FocusedFruitVeggieBuilder(output_dir="VeggiesFruits_Focused")

    # EXECUTE WITH YOUR API KEY
    builder.build_focused_dataset(
        images_per_item=150,  # 150 images per item
        pixabay_key='39098454-c5b358d4d8d9f64b822fb0ad2',
        unsplash_key=None  # Optional
    )


if __name__ == "__main__":
    main()

    # 39098454-c5b358d4d8d9f64b822fb0ad2