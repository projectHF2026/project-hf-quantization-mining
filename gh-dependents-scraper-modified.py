# gh-dependents-scraper-modified.py
"""
GitHub Dependents Scraper for Quantized Model Libraries
Optimizations over original:
  - Updated HTML parsing to handle GitHub's recent UI changes
  - Added multiple fallback methods for finding next page links
  - Improved error handling and logging
  - Added rate limit handling with dynamic sleep intervals
  - Added deduplication of results while keeping track of all source libraries
"""

import requests
import bs4
import time

LIBRARIES = [
    "huggingface/transformers",
    "AutoGPTQ/AutoGPTQ",
    "casper-hansen/AutoAWQ",
    "abetlen/llama-cpp-python",
    "TimDettmers/bitsandbytes",
    "huggingface/optimum",
    "marella/ctransformers",
    "ggerganov/llama.cpp",
    "PanQiWei/AutoGPTQ",
]

def scrape_dependents(library_name, output_file, headers):
    """Scrape GitHub dependents for a given library"""
    
    print(f"\n{'='*60}")
    print(f"Scraping dependents for: {library_name}")
    print(f"{'='*60}\n")
    
    n_reqs = 0
    dependents_count = 0
    page_num = 1
    
    URL = f"https://github.com/{library_name}/network/dependents"
    
    with open(output_file, 'a') as f:
        
        while True:
            print(f"Fetching page {page_num}...")
            response = requests.request("GET", URL, headers=headers)
            n_reqs += 1
            time.sleep(3)
            
            if response.status_code != 200:
                print(f"Error: Status code {response.status_code}")
                break
            
            soup = bs4.BeautifulSoup(response.text, 'html.parser')
            
            # Find repository links (updated selectors)
            # Try multiple possible selectors
            repo_links = soup.find_all('a', {'data-hovercard-type': 'repository'})
            
            if not repo_links:
                # Fallback to old method
                repo_links = soup.find_all('a', class_='text-bold')
            
            if not repo_links:
                print(f"No dependents found on page {page_num}")
                break
            
            # Extract repo paths and metadata
            page_count = 0
            for link in repo_links:
                repo_path = link.get('href')
                if repo_path and repo_path.startswith('/') and repo_path.count('/') >= 2:
                    # Get stars and forks from parent elements
                    parent = link.find_parent('div', class_='Box-row')
                    if parent:
                        # Try to extract stars/forks (they might not always be present)
                        stars = "0"
                        forks = "0"
                        
                        star_elem = parent.find('svg', {'aria-label': 'star'})
                        if star_elem and star_elem.parent:
                            stars = star_elem.parent.get_text(strip=True).replace(',', '')
                        
                        fork_elem = parent.find('svg', {'aria-label': 'fork'})
                        if fork_elem and fork_elem.parent:
                            forks = fork_elem.parent.get_text(strip=True).replace(',', '')
                        
                        f.write(f"{repo_path},{stars},{forks},{library_name}\n")
                        dependents_count += 1
                        page_count += 1
            
            print(f"Page {page_num}: Found {page_count} dependents (Total: {dependents_count})")
            
            # Find next page link - try multiple selectors
            next_link = None
            
            # Method 1: Look for pagination links
            pagination = soup.find('div', class_='paginate-container')
            if pagination:
                next_button = pagination.find('a', string='Next')
                if next_button:
                    next_link = next_button.get('href')
            
            # Method 2: Look for "Next" button by text
            if not next_link:
                all_links = soup.find_all('a')
                for link in all_links:
                    if link.get_text(strip=True) == 'Next':
                        next_link = link.get('href')
                        break
            
            # Method 3: Look for any link with "dependents" and "after" parameter
            if not next_link:
                for link in soup.find_all('a', href=True):
                    href = link.get('href')
                    if 'dependents' in href and 'after=' in href:
                        next_link = href
                        break
            
            if not next_link:
                print(f"No more pages found. Stopping.")
                break
            
            # Update URL for next iteration
            if next_link.startswith('http'):
                URL = next_link
            else:
                URL = f"https://github.com{next_link}"
            
            page_num += 1
            
            # Safety limit
            if page_num > 1000:
                print(f"Warning: Reached page limit of 1000. Stopping.")
                break
            
            # Rate limiting
            if n_reqs % 50 == 0:
                print(f"Rate limit pause (60s) after {n_reqs} requests...")
                time.sleep(60)
    
    print(f"\nTotal dependents for {library_name}: {dependents_count}")
    print(f"Total pages scraped: {page_num}\n")
    return dependents_count


def main():
    GITHUB_TOKEN = "YOUR_TOKEN_HERE"  # Replace with your token
    
    headers = {
        "authorization": f"token {GITHUB_TOKEN}",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    output_file = "quantized_model_dependents.csv"
    
    # Create output file with header
    with open(output_file, 'w') as f:
        f.write("repository_path,stars,forks,source_library\n")
    
    total_dependents = 0
    
    for library in LIBRARIES:
        try:
            count = scrape_dependents(library, output_file, headers)
            total_dependents += count
            print(f"✓ Completed {library}: {count} dependents")
            time.sleep(10)
        except Exception as e:
            print(f"✗ Error scraping {library}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"SCRAPING COMPLETE")
    print(f"Total dependents: {total_dependents}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}\n")
    
    remove_duplicates(output_file)


def remove_duplicates(filename):
    """Remove duplicate repos but keep track of all source libraries"""
    
    repo_data = {}
    
    with open(filename, 'r') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 4:
                continue
            repo_path, stars, forks, library = parts[0], parts[1], parts[2], parts[3]
            
            if repo_path in repo_data:
                repo_data[repo_path][2].append(library)
            else:
                repo_data[repo_path] = (stars, forks, [library])
    
    dedup_file = "quantized_model_dependents_dedup.csv"
    with open(dedup_file, 'w') as f:
        f.write("repository_path,stars,forks,source_libraries\n")
        for repo, (stars, forks, libs) in repo_data.items():
            libs_str = ';'.join(libs)
            f.write(f"{repo},{stars},{forks},{libs_str}\n")
    
    print(f"Original repos: {sum(1 for _ in open(filename)) - 1}")
    print(f"Unique repos: {len(repo_data)}")
    print(f"Deduplicated results: {dedup_file}\n")


if __name__ == "__main__":
    main()