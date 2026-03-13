import urllib.request
import json
import os
import re

API_URL = "https://rosettacode.org/w/api.php"
CATEGORY = "Category:Zen_C"

def fetch_json(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Zen-C-AutoScraper/1.0'})
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode())

def main():
    print("-> Fetching tasks from Rosetta Code...")
    
    url = f"{API_URL}?action=query&list=categorymembers&cmtitle={CATEGORY}&cmlimit=500&format=json"
    data = fetch_json(url)
    pages = data['query']['categorymembers']
    
    os.makedirs("examples/rosetta", exist_ok=True)
    os.makedirs("website_out", exist_ok=True)

    for page in pages:
        title = page['title']
        pageid = page['pageid']
        
        content_url = f"{API_URL}?action=query&prop=revisions&rvprop=content&rvslots=main&pageids={pageid}&format=json"
        content_data = fetch_json(content_url)
        text = content_data['query']['pages'][str(pageid)]['revisions'][0]['slots']['main']['*']

        parts = re.split(r'==\{\{header\|Zen[ _-]?C\}\}==', text, flags=re.IGNORECASE)
        
        if len(parts) > 1:
            zen_c_section = parts[1].split('=={{header|')[0]
            
            match = re.search(r'(?:<lang[^>]*>|<syntaxhighlight[^>]*>|<highlight[^>]*>)(.*?)(?:</lang>|</syntaxhighlight>|</highlight>)', zen_c_section, re.DOTALL | re.IGNORECASE)
            
            if match:
                code = match.group(1).strip()
                safe_title = title.replace("/", "_").replace(" ", "_")
                page_url = f"https://rosettacode.org/wiki/{title.replace(' ', '_')}"
                history_url = f"{page_url}?action=history"
                
                zc_filename = f"examples/rosetta/{safe_title}.zc"
                with open(zc_filename, "w", encoding="utf-8") as f:
                    f.write(code + "\n")
                    
                md_filename = f"website_out/{safe_title}.md"
                with open(md_filename, "w", encoding="utf-8") as f:
                    f.write("+++\n")
                    f.write(f'title = "{title}"\n')
                    f.write("+++\n\n")
                    f.write(f"# {title}\n\n")
                    f.write("```zc\n")
                    f.write(code + "\n")
                    f.write("```\n\n")
                    f.write("---\n")
                    f.write(f"**Attribution:** This is a community solution for the Rosetta Code task [**{title}**]({page_url}) in Zen C.\n\n")
                    f.write(f"*This article uses material from the Rosetta Code article **{title}**, which is released under the [GNU Free Documentation License 1.3](https://www.gnu.org/licenses/fdl-1.3.html). A list of the original authors can be found in the [page history]({history_url}).*\n")
                
                print(f"-> Scraped: {title}")
            else:
                print(f"-> Found header, but NO code block in: {title}")
        else:
            print(f"-> Could not find Zen C header in: {title}")

if __name__ == "__main__":
    main()
