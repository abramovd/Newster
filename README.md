# Newster
**Newster** is a python package for news snippets clustering. Unlike a standard approach for search results representation as a list, clustering helps to group simillar items to clusters. User can save some time by checking items in one cluster instead of scrolling the full list. 

This package is a convinient tool for searching news on such popular websites as The Guardian and The New York Times and cluster results with some well known text clustering algorithms. Besides, you can easily add some other online sources which have JSON API.

### Supported Algorithms

Now newster supports next clustering algorithms:

* K-Means Clustering
* Ward's Hierarchical Clustering Method
* Suffix Tree Clustering
* Formal Concept Analysis Algorithm Based on Probability Index - for finding the most simillar 2-3 items

### Installation

In order to install Newster on your local machine you need to complete the following steps in your terminal.

#### _Step 1_

Clone this git repo there:

```sh
$ git clone https://github.com/abramovd/Newster.git
```
Now you have ```Newster``` folder in your current directory, move there:
```sh
$ cd Newster
```

#### _Step 2_
Newster depends on Numpy, Scipy, Scikit-learn and NLTK packages. So, you need to install all the dependencies listed in ```requirements.txt``` with Pip:

```sh
$ pip install -r requirements.txt
```

#### _Step 3_

In order to query online newspapers. You need to get your own API keys on [the Guardian](http://open-platform.theguardian.com/access/) or/and [the New York Times](http://developer.nytimes.com/docs/reference/keys) websites. For NYT you need to register for Search Articles API.

## Usage

Newster package consists of two main parts: **Scraper** and **Newster** by itself.

### Scraper
The example below shows how to use Scraper to find news on the New York Times and The Guardian and work with the result.

Firstly, you need to specify your API urls and keys in two lists, e.g.:

```python
guardURL = 'http://content.guardianapis.com/search?' # Guardian URL
nytURL = 'http://api.nytimes.com/svc/search/v2/articlesearch.json?' # NYT URL
key_g = '' # #insert your Guardian api-key
key_nyt = '' # #insert your NYT api-key

api_urls = [guardURL, nytURL]
api_keys = [key_g, key_nyt]
```
Now you can create an object of class Scraper and search articles for some query:

```python
from newster.Scraper import Scraper
nyt_scraper = Scraper(nytURL, key_nyt)
query = "Obama"
```
```search()``` will return the result in JSON and save it in the object.
```python
nyt_scraper.search(query)
```
```fields()``` will return the fields of JSON:
```python
nyt_scraper.fields()
```
```python
[u'type_of_material', u'blog', u'news_desk', u'lead_paragraph', u'headline', u'abstract', u'print_page', u'word_count', u'_id', u'snippet', u'source', u'slideshow_credits', u'web_url', u'multimedia', u'subsection_name', u'keywords', u'byline', u'document_type', u'pub_date', u'section_name']
```
```show_result_by_fields(fields)``` will show the result by specified fields (list of fields or one field as a string)
```python
fields = ['word_count', 'snippet', 'web_url']
nyt_scraper.show_result_by_fields(fields)
```
or:
```python
nyt_scraper.show_result_by_fields('snippet')
```
You will have something like that:
```
word_count: 304
snippet: A federal appeals court ruling blocked the president’s plan to provide work permits to as many as five million undocumented immigrants while shielding most of them from deportation.
web_url: http://www.nytimes.com/2015/11/11/us/politics/supreme-court-immigration-obama.html
------------------------
word_count: 1622
snippet: A career policy maker takes a historical look at Middle Eastern geopolitics.
web_url: http://www.nytimes.com/2015/10/25/books/review/doomed-to-succeed-by-dennis-ross.html
.......
```
```get_result_by_field(field)``` will return a list where every element is a specified field for every search result.

As you can see, Scraper support only one news source to work with. But there is a function ```search_articles(URLs, keys, query)``` which can search articles for your query on a composition of news sources (now it supports only the Guardian and the New York Times, but some other sources can be easily added):

```
from newster.Scraper import search_articles
result = search_articles(api_urls, api_keys, "Obama")
```
It will return a dictionary in the following format: ```{'sources' : [], 'snippets', 'titles': [], 'links': []}```, where sources are NYT or GUARD.

So, then you can just use ```results['snippets']``` to see all the snippets and ```results['snippets'][3]``` to see the snippet of 4th found article. Of course, ```results['titles'][3]``` will be the title of this article.

### Newster

Newster depends on Scraper, but its mission is to cluster the results of the Scraper. Let's see the example assuming that A PI URLs and keys have been already provided:

```python
from newster.base import Newster
query = "obama"
newster = Newster(api_urls, api_keys, query)
if len(newster.get_snippets()) > 0:
    print("--------------STC---------------")
    newster.find_clusters(method = "stc", n_clusters = 6)
    newster.print_clusters()
    print("--------------FCA---------------")
    newster.find_clusters(method = "fca", n_clusters = 4)
    newster.print_clusters()
    print("-------------KMeans-------------")
    newster.find_clusters(method = "kmeans", n_clusters = 6)
    newster.print_clusters()
    print("--------------Ward---------------")
    newster.find_clusters(method = "ward", n_clusters = 6)
    newster.print_clusters()
```
The result will be the following:
```
--------------STC---------------
cluster #1 contains documents:  [19]
cluster #2 contains documents:  [3, 12]
cluster #3 contains documents:  [8, 11]
cluster #4 contains documents:  [19]
cluster #5 contains documents:  [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
cluster #6 contains documents:  [0, 2, 3, 4, 5, 6, 7, 8, 9, 12]
--------------FCA---------------
cluster #1 contains documents:  [1, 2]
cluster #2 contains documents:  [3, 12]
cluster #3 contains documents:  [8, 3]
cluster #4 contains documents:  [4, 6]
-------------KMeans-------------
cluster #1 contains documents:  [16]
cluster #2 contains documents:  [0, 3, 7, 8, 10, 12, 13, 14, 15]
cluster #3 contains documents:  [2, 6]
cluster #4 contains documents:  [4, 5, 9]
cluster #5 contains documents:  [1, 11]
cluster #6 contains documents:  [17, 18, 19]
--------------Ward---------------
cluster #1 contains documents:  [3, 7, 12, 13]
cluster #2 contains documents:  [8, 11, 15, 16]
cluster #3 contains documents:  [0, 4, 5, 6, 9]
cluster #4 contains documents:  [1, 2]
cluster #5 contains documents:  [17, 18, 19]
cluster #6 contains documents:  [10, 14]
```
Besides of ```find_clusters(method, n_clusters)``` and ```print_clusters()``` there are other important Newster's methods:

* ```search(query)``` - if a query doesn't specified in object initialization you can provide it later
* ```get_snippets()```  / ```print_snippets()```- returns / prints snippets of found articles
* ```get_links()``` / ```print_links()``` - returns / printweb-urls of found articles
* ```get_sources()``` / ```print_sources()``` - returns / prints sources of found articles (currently: NYT or GUARD)
* ```get_titles()``` / ```print_titles()``` - returns / prints titles of found articles
* ```get_clusters()``` - returns found clusters as a dictionary, e.g.: {1: [1, 2, 3], 2: [4, 5]} means 2 clusters, 1 contaiss first 3 articles and the second one - 4th and 5th articles.
* ```get_common_tags(num)``` - returns tags for clusters as dict where key is a number of a clusters and the value is a list of tags for the article (num - max number of tags per cluster)
* ```get_number_of_good_clusters``` - return number of cluster in which Suffix Tree Clustering algorithms is "sure" (works only for the STC algorithms)

Besides you can just import algorithms and use them separetely from Newster:

```python
from newster.algorithms.Ward import HierarchicalClustering
from newster.algorithms.STC import SuffixTreeClustering
from newster.algorithms.FCA import FCAClustering
from newster.algorithms.KMeans import kMeansClustering
```

## Online Implementation
**Newster Online** is an online implementation of this package. It's deployed on Heroku Server: http://newster2.herokuapp.com. For more information check this [github repository](https://github.com/abramovd/Newster-Online).

## Author
Dmitry Abramov &copy;