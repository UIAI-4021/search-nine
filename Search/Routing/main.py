import networkx as nx
import pandas as pd
import math
import heapq

from Search.Routing.Airport import Airport
from Search.Routing.Flight import Flight


def dijkstra(graph: nx.DiGraph, start: Airport, end: Airport) -> list:
    distances = {node: float('inf') for node in graph.nodes()}
    distances[start] = 0

    previous_nodes = {node: None for node in graph.nodes()}

    unvisited_nodes = set(graph.nodes())

    while unvisited_nodes:

        current_node = min(unvisited_nodes, key=lambda node: distances[node])

        unvisited_nodes.remove(current_node)

        if current_node == end:
            path = []
            while previous_nodes[current_node] is not None:
                path.append(current_node)
                current_node = previous_nodes[current_node]
            path.append(start)
            path.reverse()
            return path

        for neighbor in graph.neighbors(current_node):
            distance = distances[current_node] + graph[current_node][neighbor]['weight'].cost_edge
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
    return None


if __name__ == '__main__':

    # read dateset
    dataframe = pd.read_csv('Flight_Data.csv')

    # create graph
    graph = nx.DiGraph()

    for index, row in dataframe.iterrows():
        # create node
        source = Airport(row['SourceAirport'], row['SourceAirport_City'], row['SourceAirport_Country'],
                         row['SourceAirport_Latitude'], row['SourceAirport_Longitude'], row['SourceAirport_Altitude'])
        destination = Airport(row['DestinationAirport'], row['DestinationAirport_City'], row['DestinationAirport_Country'],
                              row['DestinationAirport_Latitude'], row['DestinationAirport_Longitude'], row['DestinationAirport_Altitude'])
        graph.add_nodes_from([source, destination])

        # create edge
        flight = Flight(row['Airline'], row['Distance'], row['FlyTime'], row['Price'])

        graph.add_weighted_edges_from([(source, destination, flight)])

    source_inp, destination_inp = input().split(" - ")

    check_source, check_destination = False, False
    for vertex in graph.nodes:
        if check_source is True and check_destination is True:
            break
        elif vertex.airport == source_inp:
            source_vertex = vertex
            check_source = True
        elif vertex.airport == destination_inp:
            destination_vertex = vertex
            check_destination = True

    best = dijkstra(graph, source_vertex, destination_vertex)
    for airport in best:
        print(airport.airport)


def haversine(latitude_source, longitude_source, latitude_destination, longitude_destination):
    earth_r = 6371

    latitude_source = math.radians(latitude_source)
    longitude_source = math.radians(longitude_source)
    latitude_destination = math.radians(latitude_destination)
    longitude_destination = math.radians(longitude_destination)

    distance_latitude = latitude_destination - latitude_source
    distance_longitude = longitude_destination - longitude_source
    a = math.sin(distance_latitude/2)**2 + (math.cos(latitude_source) *
        math.cos(latitude_destination) * math.sin(distance_longitude/2)**2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return earth_r * c


def a_stare(graph, start, goal):
    open_list = []
    closed_set = set()

    heapq.heappush([start, 0 + haversine(start.latitude, start.longitude,
                                         goal.latitude, goal.longitude)])

    while open_list:
        current_node = heapq.heappop(open_list)
        if current_node[0] == goal:
            return None # ناقض

        if current_node in closed_set:
            continue

        closed_set.add(current_node)
        # for neighbor, cost in graph[current_node]:
        #     if neighbor not in closed_set:
        #        neighbor_node =








