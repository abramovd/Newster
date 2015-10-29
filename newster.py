from algorithms.Ward import HierarchicalClustering
from Scraper import Scraper, search_articles
from config import api_urls, api_keys

query = "obama"
    
snippets = search_articles(api_urls, api_keys, query)
if len(snippets) != 0:
    hc = HierarchicalClustering(snippets)
    hc.find_clusters()
    hc.print_clusters()