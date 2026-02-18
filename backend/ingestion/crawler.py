import os
import json
import re
import asyncio
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

def clean_text(text: str) -> str:
    """
    Normalizes whitespace and removes extra newlines to ensure clean text for embedding.
    """
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def sanitize_filename(name: str) -> str:
    """
    Sanitizes a URL string to be safe for use as a directory name on the filesystem.
    Replaces special characters with underscores.
    """
    return re.sub(r'[<>:"/\\|?*]', '_', name)

def get_robots_txt(url: str) -> dict:
    """
    Fetches and parses the robots.txt file for a given URL.
    
    This is important for ethical crawling to understand which paths are allowed or disallowed.
    
    Args:
        url (str): The target URL.
        
    Returns:
        dict: A dictionary containing lists of 'Allow' and 'Disallow' paths.
    """
    parsed_url = urlparse(url)
    robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
    rules = {"Allow": [], "Disallow": []}
    
    try:
        from urllib.request import urlopen
        with urlopen(robots_url, timeout=5) as response:
            content = response.read().decode("utf-8")
            for line in content.splitlines():
                if line.strip().lower().startswith("allow:"):
                    rules["Allow"].append(line.split(":", 1)[1].strip())
                elif line.strip().lower().startswith("disallow:"):
                    rules["Disallow"].append(line.split(":", 1)[1].strip())
    except Exception as e:
        print(f"Could not fetch robots.txt: {e}")
    
    return rules

async def crawl_url(url: str) -> str:
    """
    Asynchronously crawls a URL using Playwright (Headless Browser).
    
    Why Playwright?
    Many modern websites (SPAs) load content dynamically via JavaScript. Simple requests
    libraries cannot see this content. Playwright renders the full DOM.
    
    Process:
    1. Creates a local directory named after the URL.
    2. Fetches robots.txt rules.
    3. Launches headless Chromium, navigates to page, waits for network idle.
    4. Extracts title, meta description, and cleaned body text.
    5. Saves raw text to `content.txt` and metadata to `metadata.json`.
    
    Args:
        url (str): The target URL to crawl.
        
    Returns:
        str: The cleaned extracted text content.
    """
    try:
        # Create directory
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        path = parsed_url.path.strip("/")
        folder_name = sanitize_filename(f"{domain}_{path}" if path else domain)
        output_dir = os.path.join("data", "crawled_docs", folder_name)
        os.makedirs(output_dir, exist_ok=True)

        # Fetch robots.txt (can be sync, it's fast)
        robots_rules = get_robots_txt(url)

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            # Navigate
            await page.goto(url, timeout=60000)
            
            # Wait for content
            try:
                await page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                pass
            
            content = await page.content()
            title = await page.title()
            
            description = ""
            try:
                meta_desc = page.locator('meta[name="description"]')
                if await meta_desc.count() > 0:
                    description = await meta_desc.get_attribute("content")
            except:
                pass

            await browser.close()
            
            soup = BeautifulSoup(content, 'html.parser')
            for script in soup(["script", "style", "header", "footer", "nav", "noscript"]):
                script.decompose()
                
            text = soup.get_text(separator=' ')
            cleaned_text = clean_text(text)
            
            # Save files
            with open(os.path.join(output_dir, "content.txt"), "w", encoding="utf-8") as f:
                f.write(cleaned_text)
                
            metadata = {
                "url": url,
                "title": title,
                "description": description,
                "robots_rules": robots_rules
            }
            with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
                
            print(f"Saved crawled data to {output_dir}")
            return cleaned_text
            
    except Exception as e:
        print(f"Error crawling URL {url}: {e}")
        return ""
