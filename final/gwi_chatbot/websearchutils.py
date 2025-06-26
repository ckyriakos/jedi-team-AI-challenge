import re
from typing import List, Dict, Any

def format_web_search_citation(result: Dict[str, Any], index: int) -> str:
    """Format a web search result as a citation"""
    title = result.get('title', f'Web Result {index + 1}')
    url = result.get('url', '#')
    snippet = result.get('snippet', result.get('description', ''))
    
    # Clean up the snippet
    if snippet:
        snippet = snippet[:200] + "..." if len(snippet) > 200 else snippet
        snippet = re.sub(r'\s+', ' ', snippet).strip()
    
    citation_text = f"<strong>Source {index + 1}:</strong> <a href='{url}' target='_blank'>{title}</a>"
    if snippet:
        citation_text += f"<br><em>Excerpt:</em> {snippet}"
    
    return citation_text

def extract_citations_from_text(text: str) -> List[str]:
    """Extract citation markers from text (e.g., [1], [Source 1])"""
    citation_pattern = r'\[(?:Source\s+)?(\d+)\]'
    citations = re.findall(citation_pattern, text, re.IGNORECASE)
    return list(set(citations))  # Remove duplicates

def add_citation_numbers_to_response(response: str, sources: List[Dict]) -> str:
    """Add citation numbers to response text where sources are referenced"""
    # This is a simple implementation - you might want to make it more sophisticated
    modified_response = response
    
    # Look for phrases that might indicate source usage
    source_indicators = [
        r'according to',
        r'based on',
        r'as mentioned in',
        r'the document states',
        r'research shows',
        r'studies indicate'
    ]
    
    for i, indicator in enumerate(source_indicators):
        pattern = f'({indicator})'
        replacement = f'\\1 [Source {(i % len(sources)) + 1}]'
        modified_response = re.sub(pattern, replacement, modified_response, flags=re.IGNORECASE)
    
    return modified_response

def create_bibliography(sources: List[Dict]) -> str:
    """Create a formatted bibliography from sources"""
    bibliography = "\n**Sources:**\n"
    
    for i, source in enumerate(sources):
        if 'url' in source:  # Web search result
            title = source.get('title', f'Web Source {i + 1}')
            url = source.get('url', '#')
            bibliography += f"{i + 1}. [{title}]({url})\n"
        else:  # Document source
            metadata = source.get('metadata', {})
            source_name = metadata.get('source', f'Document {i + 1}')
            page = metadata.get('page', '')
            bibliography += f"{i + 1}. {source_name}"
            if page:
                bibliography += f" (Page {page})"
            bibliography += "\n"
    
    return bibliography
