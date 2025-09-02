"""
PubMed API + LlamaIndex Demo
Shortest demo showing how to:
1. Search PubMed for papers
2. Create LlamaIndex documents from abstracts
3. Query with OpenAI
"""

import os
import argparse
import re
import csv
import json
from dotenv import load_dotenv
from Bio import Entrez
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Load environment variables
load_dotenv()

# Configure LlamaIndex
Settings.llm = OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
Settings.embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"))

# Set email for PubMed (required by NCBI)
Entrez.email = "your.email@example.com"

def load_config(config_file="config.json"):
    """Load configuration from JSON file"""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"⚠️  Config file {config_file} not found. Using default settings.")
        return get_default_config()
    except json.JSONDecodeError as e:
        print(f"⚠️  Error parsing config file: {e}. Using default settings.")
        return get_default_config()

def get_default_config():
    """Return default configuration"""
    return {
        "search_config": {
            "filter_journals": False,
            "query_index": 0,
            "top_journals": [
                '"J Am Med Inform Assoc"[Journal]',
                '"Nat Med"[Journal]',
                '"Lancet Digit Health"[Journal]'
            ],
            "queries": ["machine learning AND mental health"],
            "target_regions": ["none"],
            "max_results": 100
        },
        "export_config": {
            "filename": "papers.csv",
            "auto_export": False
        },
        "analysis_config": {
            "batch_query": 5,
            "default_queries": [
                "What are the main topics covered in these papers?",
                "What methodologies are commonly used?"
            ]
        }
    }

def extract_university_and_department(affiliation_text):
    """
    Extract university and department names separately from PubMed affiliation text
    Returns: (university, department) tuple
    """
    # Clean up the original text
    affiliation = affiliation_text.strip()
    
    # Remove common suffixes first
    cleanup_patterns = [
        r',?\s*\b\d{5}[-\s]?\d{0,4}\b.*$',  # US zip codes
        r',?\s*\b[A-Z]{2}\s*\d{5}.*$',      # State + zip
        r',?\s*\b(?:USA|US|United States|UK|Canada|Australia)\b.*$',
        r'\.?\s*(?:Electronic address:|E-mail:|Email:).*$',
        r',?\s*\w+@\w+\.\w+.*$',  # Email addresses
    ]
    
    for pattern in cleanup_patterns:
        affiliation = re.sub(pattern, '', affiliation, flags=re.IGNORECASE)
    
    # Split by common separators
    parts = [part.strip() for part in re.split(r'[,;]', affiliation) if part.strip()]
    
    university = "N/A"
    department = "N/A"
    
    # Look for university/college/hospital in parts
    university_patterns = [
        r'.*(?:University|College|Medical Center|Hospital|Institute of Technology|Health System).*',
        r'.*(?:Universit[yé]|Institut).*'
    ]
    
    # Look for department/school patterns
    department_patterns = [
        r'.*(?:Department|School|Division|Center|Laboratory|Lab).*',
        r'.*(?:Faculty|College) of.*'
    ]
    
    for part in parts:
        # Check if this part is a university
        for pattern in university_patterns:
            if re.match(pattern, part, re.IGNORECASE) and university == "N/A":
                university = part
                break
        
        # Check if this part is a department
        for pattern in department_patterns:
            if re.match(pattern, part, re.IGNORECASE) and department == "N/A":
                department = part
                break
    
    # If we found both, we're done
    if university != "N/A" and department != "N/A":
        return university, department
    
    # If we only found university, try to extract department from the same part
    if university != "N/A" and department == "N/A":
        # Look for department within the university string
        dept_in_univ = re.search(r'(Department|School|Division|Center|Laboratory|Lab|Faculty|College)\s+of\s+[^,;]+', university, re.IGNORECASE)
        if dept_in_univ:
            department = dept_in_univ.group(0)
            # Remove department from university name
            university = re.sub(re.escape(dept_in_univ.group(0)), '', university).strip(' ,-')
    
    # Fallback: if no clear university found, use the longest meaningful part
    if university == "N/A" and parts:
        university = max(parts, key=len)
    
    return university, department

def filter_papers_by_regions_with_llm(documents, target_regions, batch_size=5):
    """Use LLM to filter papers by target regions in batches"""
    if not target_regions or (len(target_regions) == 1 and target_regions[0].lower() == "none"):
        return documents
        
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  No OpenAI API key found. Skipping region filtering.")
        return documents
    
    regions_str = ", ".join(target_regions)
    print(f"🌍 Using LLM to filter for {regions_str}-affiliated papers (batch size: {batch_size})...")
    
    # Initialize LLM for filtering
    from llama_index.llms.openai import OpenAI
    llm = OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    
    filtered_papers = []
    
    # Extract institutions for batch processing
    institutions_to_check = []
    doc_institution_map = {}
    
    for doc in documents:
        institution = ""
        if doc.metadata and 'corresponding_author' in doc.metadata:
            if doc.metadata['corresponding_author'] and 'university' in doc.metadata['corresponding_author']:
                institution = doc.metadata['corresponding_author']['university']
        
        if institution:
            institutions_to_check.append(institution)
            doc_institution_map[institution] = doc
    
    # Process institutions in batches
    for i in range(0, len(institutions_to_check), batch_size):
        batch = institutions_to_check[i:i+batch_size]
        
        # Create batch prompt
        institution_list = "\n".join([f"{j+1}. {inst}" for j, inst in enumerate(batch)])
        
        prompt = f"""
        Analyze these institution names and determine which ones are located in any of these regions: {regions_str}.
        
        Institutions:
        {institution_list}
        
        Consider:
        - University names, medical centers, hospitals in {regions_str}
        - Geographic indicators for these regions
        - Common institutional patterns for these regions
        
        Respond with only the numbers (1, 2, 3, etc.) of institutions that are clearly in ANY of these regions: {regions_str}, separated by commas.
        If none are in these regions, respond with "NONE".
        Example response: "1, 3, 5" or "NONE"
        """
        
        try:
            response = llm.complete(prompt)
            result = response.text.strip()
            
            if result.upper() == "NONE":
                keep_indices = []
            else:
                # Parse the response to get indices
                keep_indices = []
                for num_str in result.split(','):
                    try:
                        idx = int(num_str.strip()) - 1  # Convert to 0-based index
                        if 0 <= idx < len(batch):
                            keep_indices.append(idx)
                    except ValueError:
                        continue
            
            # Process results
            for j, institution in enumerate(batch):
                if j in keep_indices:
                    filtered_papers.append(doc_institution_map[institution])
                    print(f"✅ Keeping: {institution}")
                else:
                    print(f"❌ Filtering out: {institution}")
                    
        except Exception as e:
            print(f"⚠️  Error processing batch {i//batch_size + 1}: {e}")
            # Keep all papers in batch if LLM fails
            for institution in batch:
                filtered_papers.append(doc_institution_map[institution])
    
    print(f"🌍 Filtered to {len(filtered_papers)} papers from target regions ({len(documents)} total)")
    return filtered_papers

def export_papers_to_csv(documents, filename="usa_papers.csv", analysis_config=None):
    """Export papers to CSV file with university, department, abstract, and AI analysis columns"""
    print(f"📄 Exporting {len(documents)} papers to {filename}...")
    
    # Get analysis queries
    analysis_queries = []
    if analysis_config and "default_queries" in analysis_config:
        analysis_queries = analysis_config["default_queries"]
    
    # Create fieldnames dynamically based on analysis queries
    fieldnames = ['name', 'PMID', 'corresponding_author', 'university', 'department', 'abstract']
    for i, query in enumerate(analysis_queries):
        fieldnames.append(f'analysis_{i+1}')
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header
        writer.writeheader()
        
        # Prepare data for batch analysis
        papers_data = []
        papers_metadata = []
        
        for doc in documents:
            # Extract data from document metadata
            title = doc.metadata.get('title', 'N/A') if doc.metadata else 'N/A'
            pmid = doc.metadata.get('pmid', 'N/A') if doc.metadata else 'N/A'
            abstract = doc.metadata.get('abstract', 'N/A') if doc.metadata else 'N/A'
            
            corresponding_author = 'N/A'
            university = 'N/A'
            department = 'N/A'
            
            if doc.metadata and 'corresponding_author' in doc.metadata:
                if doc.metadata['corresponding_author']:
                    corresponding_author = doc.metadata['corresponding_author'].get('name', 'N/A')
                    university = doc.metadata['corresponding_author'].get('university', 'N/A')
                    department = doc.metadata['corresponding_author'].get('department', 'N/A')
            
            papers_data.append((title, abstract))
            papers_metadata.append({
                'title': title,
                'pmid': pmid,
                'corresponding_author': corresponding_author,
                'university': university,
                'department': department,
                'abstract': abstract
            })
        
        # Get AI analysis results in batches
        all_analysis_results = []
        if analysis_queries:
            print(f"🤖 Analyzing {len(papers_data)} papers in batches of 10...")
            all_analysis_results = analyze_papers_batch_with_llm(papers_data, analysis_queries, batch_size=10)
        else:
            all_analysis_results = [["N/A"] * len(analysis_queries) for _ in papers_data]
        
        # Write paper data
        for i, metadata in enumerate(papers_metadata):
            analysis_results = all_analysis_results[i] if i < len(all_analysis_results) else ["N/A"] * len(analysis_queries)
            
            # Create row data
            row_data = {
                'name': metadata['title'],
                'PMID': metadata['pmid'],
                'corresponding_author': metadata['corresponding_author'],
                'university': metadata['university'],
                'department': metadata['department'],
                'abstract': metadata['abstract']
            }
            
            # Add analysis results
            for j, result in enumerate(analysis_results):
                row_data[f'analysis_{j+1}'] = result
            
            writer.writerow(row_data)
    
    print(f"✅ Successfully exported to {filename}")
    return filename

def analyze_papers_batch_with_llm(papers_data, queries, batch_size=10):
    """Analyze multiple papers in batches using LLM"""
    if not os.getenv("OPENAI_API_KEY"):
        return [["N/A"] * len(queries) for _ in papers_data]
    
    from llama_index.llms.openai import OpenAI
    llm = OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    
    all_results = []
    
    # Process papers in batches
    for i in range(0, len(papers_data), batch_size):
        batch = papers_data[i:i+batch_size]
        batch_results = []
        
        for query in queries:
            # Create batch prompt for all papers in this batch
            papers_text = ""
            for j, (title, abstract) in enumerate(batch):
                if abstract and abstract != 'N/A':
                    papers_text += f"\nPaper {j+1}: {title}\nAbstract: {abstract}\n"
                else:
                    papers_text += f"\nPaper {j+1}: {title}\nAbstract: No abstract available\n"
            
            prompt = f"""
            Analyze these research papers and answer the question for each paper with ONLY ONE WORD per paper.
            
            {papers_text}
            
            Question: {query}
            
            Instructions:
            - Respond with exactly ONE word per paper
            - Format: "1: word1, 2: word2, 3: word3" etc.
            - Choose the most relevant single word that answers the question
            - If unclear, respond with "Unknown"
            
            One word answers:"""
            
            try:
                response = llm.complete(prompt)
                result_text = response.text.strip()
                
                # Parse the batch response
                query_results = []
                for j in range(len(batch)):
                    # Look for pattern "j+1: word"
                    import re
                    pattern = f"{j+1}:\\s*([^,\\n]+)"
                    match = re.search(pattern, result_text)
                    if match:
                        word = match.group(1).strip().split()[0]  # Take first word only
                        query_results.append(word)
                    else:
                        query_results.append("Unknown")
                
                batch_results.append(query_results)
                
            except Exception as e:
                print(f"⚠️  Error analyzing batch: {e}")
                batch_results.append(["Error"] * len(batch))
        
        # Transpose batch_results to get results per paper
        for j in range(len(batch)):
            paper_results = [batch_results[q][j] for q in range(len(queries))]
            all_results.append(paper_results)
        
        print(f"🤖 Analyzed batch {i//batch_size + 1}/{(len(papers_data) + batch_size - 1)//batch_size}")
    
    return all_results

def analyze_paper_with_llm(abstract, queries):
    """Analyze a single paper abstract using LLM with default queries"""
    if not os.getenv("OPENAI_API_KEY") or not abstract or abstract == 'N/A':
        return ["N/A"] * len(queries)
    
    from llama_index.llms.openai import OpenAI
    llm = OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    
    results = []
    
    for query in queries:
        prompt = f"""
        Read this research paper abstract and answer the question with ONLY ONE WORD.
        
        Abstract: "{abstract}"
        
        Question: {query}
        
        Instructions:
        - Respond with exactly ONE word only
        - Choose the most relevant single word that answers the question
        - If unclear, respond with "Unknown"
        
        One word answer:"""
        
        try:
            response = llm.complete(prompt)
            result = response.text.strip().split()[0]  # Take only first word
            results.append(result)
        except Exception as e:
            print(f"⚠️  Error analyzing paper: {e}")
            results.append("Error")
    
    return results

def print_detailed_papers(documents, count=5, analysis_config=None):
    """Print detailed information for first N papers with AI analysis"""
    print(f"\n📋 Detailed Information for First {min(count, len(documents))} Papers:")
    print("=" * 80)
    
    # Get analysis queries if provided
    analysis_queries = []
    if analysis_config and "default_queries" in analysis_config:
        analysis_queries = analysis_config["default_queries"]
    
    for i, doc in enumerate(documents[:count]):
        print(f"\n📄 Paper {i+1}:")
        print("-" * 40)
        
        # Extract metadata
        title = doc.metadata.get('title', 'N/A') if doc.metadata else 'N/A'
        pmid = doc.metadata.get('pmid', 'N/A') if doc.metadata else 'N/A'
        abstract = doc.metadata.get('abstract', 'N/A') if doc.metadata else 'N/A'
        first_author = doc.metadata.get('first_author', 'N/A') if doc.metadata else 'N/A'
        
        print(f"📝 Title: {title}")
        
        # AI Analysis results as second row
        if analysis_queries and abstract != 'N/A':
            print("🤖 AI Analysis:", end=" ")
            analysis_results = analyze_paper_with_llm(abstract, analysis_queries)
            print(" | ".join(analysis_results))
        
        print(f"🆔 PMID: {pmid}")
        print(f"👤 First Author: {first_author}")
        
        if doc.metadata and 'corresponding_author' in doc.metadata:
            if doc.metadata['corresponding_author']:
                corr_author = doc.metadata['corresponding_author']
                print(f"📧 Corresponding Author: {corr_author.get('name', 'N/A')}")
                print(f"🏫 University: {corr_author.get('university', 'N/A')}")
                if corr_author.get('department', 'N/A') != 'N/A':
                    print(f"🏛️ Department: {corr_author.get('department', 'N/A')}")
        
        print(f"📄 Abstract: {abstract[:300]}{'...' if len(abstract) > 300 else ''}")
        print()

def search_pubmed(query, max_results=10):
    """Search PubMed and return paper details"""
    print(f"🔍 Searching PubMed for: '{query}'")
    
    # Search PubMed
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    search_results = Entrez.read(handle)
    handle.close()
    
    # Get paper details
    ids = search_results["IdList"]
    if not ids:
        print("No papers found!")
        return []
    
    # Fetch detailed XML format to get more author info
    handle = Entrez.efetch(db="pubmed", id=ids, rettype="xml", retmode="xml")
    papers_xml = Entrez.read(handle)
    handle.close()
    
    # Parse papers from XML
    documents = []
    
    print(f"\n📋 Found Papers:")
    print("-" * 70)
    
    for i, article in enumerate(papers_xml['PubmedArticle'][:max_results]):
        try:
            # Extract basic info
            medline_citation = article['MedlineCitation']
            pmid = str(medline_citation['PMID'])
            
            # Title
            title = medline_citation['Article']['ArticleTitle']
            
            # Abstract
            abstract = ""
            if 'Abstract' in medline_citation['Article']:
                abstract_texts = medline_citation['Article']['Abstract']['AbstractText']
                if isinstance(abstract_texts, list):
                    abstract = " ".join([str(text) for text in abstract_texts])
                else:
                    abstract = str(abstract_texts)
            
            # Authors and corresponding author info
            corresponding_author = None
            first_author = None
            
            if 'AuthorList' in medline_citation['Article']:
                authors = medline_citation['Article']['AuthorList']
                
                for j, author in enumerate(authors):
                    if 'LastName' in author and 'ForeName' in author:
                        author_name = f"{author['ForeName']} {author['LastName']}"
                        
                        if j == 0:  # First author
                            first_author = author_name
                        
                        # Check for corresponding author indicators
                        if 'AffiliationInfo' in author:
                            for affiliation in author['AffiliationInfo']:
                                if 'Affiliation' in affiliation:
                                    affil_text = affiliation['Affiliation']
                                    university, department = extract_university_and_department(affil_text)
                                    
                                    if university != "N/A":
                                        corresponding_author = {
                                            'name': author_name,
                                            'university': university,
                                            'department': department
                                        }
                                        break
            
            # Display paper info
            print(f"{i+1}. {title}")
            print(f"   PMID: {pmid}")
            
            if first_author:
                print(f"   First Author: {first_author}")
            
            if corresponding_author:
                print(f"   Corresponding Author: {corresponding_author['name']}")
                print(f"   University: {corresponding_author['university']}")
                if corresponding_author['department'] != "N/A":
                    print(f"   Department: {corresponding_author['department']}")
            else:
                print(f"   ⚠️  No corresponding author institution found")
            
            if abstract:
                doc_text = f"Title: {title}\n\nAbstract: {abstract}"
                documents.append(Document(
                    text=doc_text,
                    metadata={
                        "pmid": pmid, 
                        "title": title,
                        "abstract": abstract,
                        "first_author": first_author,
                        "corresponding_author": corresponding_author
                    }
                ))
                print(f"   ✅ Abstract available")
            else:
                print(f"   ⚠️  No abstract available")
            print()
            
        except Exception as e:
            print(f"   ❌ Error parsing paper {i+1}: {e}")
            continue
    
    print(f"✅ {len(documents)} papers with abstracts ready for analysis")
    return documents

def search_with_journal_filter(base_query, top_journals, max_results=500):
    """Search PubMed with journal filtering"""
    # Combine journals with OR
    journal_filter = " OR ".join(top_journals)
    
    # Final query
    full_query = f'({base_query}) AND ({journal_filter})'
    
    return search_pubmed(full_query, max_results=max_results)

def main():
    """Main demo function with JSON configuration support"""
    parser = argparse.ArgumentParser(description="PubMed + LlamaIndex Demo")
    parser.add_argument("--analyze", action="store_true", 
                       help="Enable AI analysis of papers (requires OpenAI API key)")
    parser.add_argument("--config", default="config.json",
                       help="Path to JSON configuration file (default: config.json)")
    args = parser.parse_args()
    
    print("🧬 PubMed + LlamaIndex Demo")
    print("=" * 35)
    
    # Load configuration
    config = load_config(args.config)
    search_config = config["search_config"]
    export_config = config["export_config"]
    analysis_config = config["analysis_config"]
    
    if args.analyze and not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY required for AI analysis!")
        return
    
    try:
        # Select query from config
        queries = search_config["queries"]
        query_index = search_config["query_index"]
        if query_index >= len(queries):
            print(f"⚠️  Query index {query_index} out of range. Using index 0.")
            query_index = 0
            
        base_query = queries[query_index]
        print(f"📝 Using query: {base_query}")
        
        # Search with or without journal filtering
        if search_config["filter_journals"]:
            print("🎯 Searching in configured top journals...")
            documents = search_with_journal_filter(
                base_query, 
                search_config["top_journals"], 
                max_results=search_config["max_results"]
            )
        else:
            print("🔍 Searching all journals...")
            documents = search_pubmed(base_query, max_results=search_config["max_results"])
        

        if not documents:
            print("No papers with abstracts found. Try a broader search.")
            return
        
        # Apply region filtering if configured
        target_regions = search_config.get("target_regions", [])
        if target_regions and not (len(target_regions) == 1 and target_regions[0].lower() == "none"):
            batch_size = analysis_config.get("batch_query", 5)
            documents = filter_papers_by_regions_with_llm(documents, target_regions, batch_size)
            if not documents:
                print(f"No papers found from target regions: {', '.join(target_regions)}")
                return
        
        # Print detailed information for first 5 papers
        print_detailed_papers(documents, count=5, analysis_config=analysis_config)
        
        # Export papers if configured
        if export_config["auto_export"]:
            export_papers_to_csv(documents, export_config["filename"], analysis_config)
        
        if not args.analyze:
            print("\n📋 Search complete! Use --analyze flag for AI analysis.")
            return
            
        # AI Analysis mode
        print("🏗️  Building vector index...")
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()
        
        # Use configured analysis queries
        analysis_queries = analysis_config["default_queries"]
        
        print("\n🤖 AI Analysis of Papers:")
        print("-" * 35)
        
        for i, question in enumerate(analysis_queries, 1):
            print('hi')
            print(f"\n{i}. {question}")
            response = query_engine.query(question)
            print(f"   Answer: {response}")
            
    except Exception as e:
        print("Error during calling openai api")

if __name__ == "__main__":
    main()