import pandas as pd
import numpy as np
import requests
import re
from bs4 import BeautifulSoup as bs
import random
import time

# searchRadius = [0, 10, 25, 50, 75, 100, 200, 300, 400, 500]

def __main__():
    query = {
        "city": "Arlington",
        "state": "VA",
        "zip": "22222",
        "searchRadius": 0,
        "makeCodeList": "MAZDA",
        "modelCodeList": "CX-5",
        "driveGroup": "AWD4WD",
        "sellerTypes": "d"
    }
    AT = AutoTrader(query)
    data = AT.get_data()
    AT.save_data()


class AutoTrader:
    def __init__(self, query):
        self.query = query
        self.firstRecord = 0
        self.numRecords = 100
        self.trim = ['Grand Touring Reserve', 'Grand Touring', 'Grand Select', 'Signature', 'Touring', 'Sport']
        city = self.query['city']
        state = self.query['state']
        zipcode = self.query['zip']
        self.base_url = "https://www.autotrader.com/cars-for-sale/{}+{}+{}".format(city, state, zipcode)
        self.headers = {
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3",
            "accept-encoding": "gzip, deflate, br",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "max-age=0",
            "dnt": "1",
            "sec-fetch-mode": "navigate",
            "sec-fetch-site": "same-origin",
            "sec-fetch-user": "?1",
            "upgrade-insecure-requests": "1",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.120 Safari/537.36"
        }

    def save_data(self):
        self.results.to_csv("{}_{}_{}.csv".format(self.query["zip"], self.query["makeCodeList"], self.query["modelCodeList"]), index=False)
        return

    def get_data(self):
        print("Get data {} - {}".format(self.firstRecord, self.firstRecord + self.numRecords))
        r = self.data_request()
        soup = bs(r.text, 'html.parser')
        result_total = soup.find("div", {"class": "results-text-container"}).get_text()
        result_total = int(re.sub("(.+ of | Results|,|[+])", "", result_total))
        car_data = self.parse_html(r)
        while self.firstRecord < result_total:
            wait_time = random.random()*30+20
            print("-Hold for {} seconds (adjust if desired but use best practices to avoid being blocked)".format(wait_time))
            time.sleep(wait_time)
            print("Get data {} - {}".format(self.firstRecord, self.firstRecord + self.numRecords))
            r = self.data_request()
            page_data = self.parse_html(r)
            car_data = np.append(car_data, page_data)
        self.results = pd.DataFrame(list(car_data))
        return self.results

    def data_request(self):
        params = {
            "searchRadius": self.query["searchRadius"],
            "sortBy": "relevance",
            "numRecords": self.numRecords,
            "firstRecord": self.firstRecord,
            "makeCodeList": self.query["makeCodeList"],
            "modelCodeList": self.query["modelCodeList"],
            "driveGroup": self.query["driveGroup"],
            "sellerTypes": self.query["sellerTypes"],
            "marketExtension": True
        }
        r = requests.get(self.base_url, params=params, headers=self.headers)
        self.firstRecord += self.numRecords
        return r

    def parse_html(self, r):
        soup = bs(r.text, 'html.parser')
        listings_group = soup.find("div", {"data-qaid": "cntnr-listings-tier-listings"})
        all_listings = listings_group.findAll("div", {"data-cmp": "inventoryListing"})
        listing_data = [self.extract_information(listing) for listing in all_listings]
        return listing_data

    def extract_information(self, listing):
        try:
            subheading = listing.find("h2", {"data-cmp": "subheading"}).get_text()
            year = int(re.search("[0-9]{4}", subheading).group())
            for t in self.trim:
                if t in subheading:
                    trim = t
                    break
        except:
            year = trim = None
        try:
            pricing = listing.find("div", {"data-cmp": "pricing"}).get_text()
            pricing = str("".join(re.findall(r"[0-9]+", pricing)))
            if len(pricing) == 10:
                pricing = int(re.findall('([0-9]{5})(?:[0-9]{5})', pricing)[0])
            elif len(pricing) == 8:
                pricing = int(re.findall('([0-9]{4})(?:[0-9]{4})', pricing)[0])
        except:
            pricing = None
        try:
            miles = listing.find("div", {"class": "item-card-specifications"}).span.get_text()
            miles = int("".join(re.findall(r"[0-9]+", miles)))
        except:
            miles = 0
        try:
            info = listing.find("ul", {"data-cmp": "list"})
            color = info.li.span.get_text()
            if "Color" in color:
                color = color.replace("Color: ", "")
            else:
                color = None
        except:
            color = None
        try:
            miles_away_text = listing.find("div", {"data-cmp": "ownerDistance"}).span.get_text()
            miles_away = float(".".join(re.findall(r"[0-9]+", miles_away_text)))
        except:
            miles_away = None
        try:
            stars = listing.find("div", {"data-cmp": "starRating"})
            star = stars.findAll("span")
            dealer_rating = 0
            for s in star:
                s_style = s.findAll("span")[1]['style']
                s_value = float(re.search("width:([0-9]+)%", s_style))
                dealer_rating += s_value
        except:
            dealer_rating = None

        return {
            "year": year,
            "trim": trim,
            "pricing": pricing,
            "miles": miles,
            "color": color,
            "miles_away": miles_away,
            "dealer_rating": dealer_rating
        }

def cluster(data):
    # clustering dataset
    # determine k using elbow method

    from sklearn.cluster import KMeans
    from sklearn import metrics
    from scipy.spatial.distance import cdist
    import numpy as np
    import matplotlib.pyplot as plt

    # create new plot and data
    plt.plot()
    colors = ['b', 'g', 'r']
    markers = ['o', 'v', 's']

    # k means determine k
    distortions = []
    K = range(1,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(data)
        kmeanModel.fit(data)
        distortions.append(sum(np.min(cdist(data, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])

    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

#clf = KMeans(n_clusters=6, n_jobs=-1, random_state=13).fit(data_unsupervised)
#pred = clf.labels_
#data['cluster'] = pred
#data.sample(5)