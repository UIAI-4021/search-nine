import networkx as nx
import pandas as pd

from Search.Routing.Airport import Airport
from Search.Routing.Flight import Flight

if __name__ == '__main__':

    # read dateset
    dataframe = pd.read_csv('Flight_Data.csv')

    # create graph
    graph = nx.Graph()

    for index, row in dataframe.iterrows():
        # create node
        source = Airport(row['SourceAirport'], row['SourceAirport_City'], row['SourceAirport_Country'])
        destination = Airport(row['DestinationAirport'], row['DestinationAirport_City'], row['DestinationAirport_Country'])
        graph.add_nodes_from([source, destination])

        # create edge
        flight = Flight(row['Airline'], row['SourceAirport'], row['DestinationAirport'], row['SourceAirport_City'],
                        row['SourceAirport_Country'], row['SourceAirport_Latitude'], row['SourceAirport_Longitude'],
                        row['SourceAirport_Altitude'], row['DestinationAirport_City'], row['DestinationAirport_Country'],
                        row['DestinationAirport_Latitude'], row['DestinationAirport_Longitude'],
                        row['DestinationAirport_Altitude'], row['Distance'], row['FlyTime'], row['Price'])

        graph.add_weighted_edges_from([(source, destination, flight)])

    print(graph)

