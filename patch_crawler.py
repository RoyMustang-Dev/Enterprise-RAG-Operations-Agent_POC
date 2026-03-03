import sys
content = open('app/ingestion/crawler_service.py', 'r', encoding='utf-8').read()

# Replace BeautifulSoup imports
content = content.replace('from bs4 import BeautifulSoup', 'from selectolax.parser import HTMLParser')
content = content.replace('import urllib.robotparser', 'import urllib.robotparser\nfrom urllib.parse import urlparse, urljoin, urldefrag, urlencode, parse_qs')

# Replace _clean_content function
old_clean = '''    def _clean_content(self, html: str) -> str:
        \"\"\"Removes headers, footers, navs to extract main content.\"\"\"
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove distractions
        for tag in soup(["header", "footer", "nav", "aside", "script", "style", "noscript", "iframe"]):
            tag.decompose()
            
        return soup.get_text(separator="\\n", strip=True)'''

new_clean = '''    def _clean_content(self, html: str) -> str:
        \"\"\"Selectolax ultra-fast string extraction.\"\"\"
        if not html: return ""
        tree = HTMLParser(html)
        
        # Remove distractions
        tags_to_remove = ["header", "footer", "nav", "aside", "script", "style", "noscript", "iframe", "svg"]
        for t in tags_to_remove:
            for node in tree.css(t):
                node.decompose()
                
        return tree.text(separator="\\n", strip=True)'''

content = content.replace(old_clean, new_clean)

# Also replace soup anchor extraction in crawler
old_soup_parse = 'soup = BeautifulSoup(content_html, "html.parser")\n                            discovered = []\n                            for a in soup.find_all("a", href=True):'
new_soup_parse = 'tree = HTMLParser(content_html)\n                            discovered = []\n                            for a in tree.css("a"): \n                                if not a.attributes.get("href"): continue \n                                a = {"href": a.attributes["href"]}'
content = content.replace(old_soup_parse, new_soup_parse)

old_soup_parse_2 = 'soup = BeautifulSoup(content_html, "html.parser")\n                    discovered = []\n\n                    for a in soup.find_all("a", href=True):'
new_soup_parse_2 = 'tree = HTMLParser(content_html)\n                    discovered = []\n\n                    for node in tree.css("a"):\n                        if not node.attributes.get("href"): continue\n                        a = {"href": node.attributes["href"]}'
content = content.replace(old_soup_parse_2, new_soup_parse_2)

open('app/ingestion/crawler_service.py', 'w', encoding='utf-8').write(content)
