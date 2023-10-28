import networkx as nx
import pandas as pd

from Search.Routing.Airport import Airport

if __name__ == '__main__':

    # read dateset
    dataframe = pd.read_csv('Flight_Data.csv')

    airports = set()

    for index, row in dataframe.iterrows():
        airports.add(Airport(row['SourceAirport'], row['SourceAirport_City'], row['SourceAirport_Country']))
        airports.add(Airport(row['DestinationAirport'], row['DestinationAirport_City'], row['DestinationAirport_Country']))

    graph = nx.Graph

    for airport in airports:
        graph.add_node(airport)

    flights = list()
    for index, row in dataframe.iterrows():
        pass

