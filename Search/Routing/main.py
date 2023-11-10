import networkx as nx
import pandas as pd
import math
import heapq
import pickle
import os
import time

from Search.Routing.Airport import Airport
from Search.Routing.Flight import Flight

earth_r = 6371
speed = 800  # km / h
fuel_per_K = 0.35  # G
price_fuel = 184  # $


class Node:
    def __init__(self, state, father, cost):
        self.state = state
        self.father = father
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost


def dijkstra(graph, start, goal):
    distances = {node: float('inf') for node in graph.nodes()}
    distances[start] = 0

    previous_nodes = {node: None for node in graph.nodes()}

    unvisited_nodes = set(graph.nodes())

    while unvisited_nodes:

        current_node = min(unvisited_nodes, key=lambda node: distances[node])

        unvisited_nodes.remove(current_node)

        if current_node == goal:
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


def heuristic(start, goal):
    latitude_source = math.radians(start.latitude)
    longitude_source = math.radians(start.longitude)
    latitude_destination = math.radians(goal.latitude)
    longitude_destination = math.radians(goal.longitude)

    distance_latitude = latitude_destination - latitude_source
    distance_longitude = longitude_destination - longitude_source
    a = math.sin(distance_latitude / 2) ** 2 + (math.cos(latitude_source) *
                                                math.cos(latitude_destination) * math.sin(distance_longitude / 2) ** 2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # distance + fly time + price --> cost
    distance = earth_r * c
    fly_time = distance / speed
    price = distance * price_fuel * fuel_per_K
    return distance + fly_time + price


def construct_path(node):
    path = []
    while node:
        path.append(node.state)
        node = node.father
    return list(reversed(path))


def a_stare(graph, start, goal):
    open_list = [Node(start, None, 0 + heuristic(start, goal))]
    close_set = set()

    while open_list:
        current_node = heapq.heappop(open_list)

        if current_node.state == goal:
            return construct_path(current_node)

        if current_node.state in close_set:
            continue

        close_set.add(current_node.state)

        for neighbor in graph[current_node.state]:
            if neighbor not in close_set:
                gn = graph[current_node.state][neighbor]['weight'].cost_edge
                hn = heuristic(current_node.state, goal)
                node = Node(neighbor, current_node, gn + hn)
                heapq.heappush(open_list, node)
    return None


if __name__ == '__main__':

    if os.path.exists('graph.pkl') is False:
        # read dateset
        dataframe = pd.read_csv('Flight_Data.csv')

        # create graph
        graph = nx.DiGraph()

        for index, row in dataframe.iterrows():
            # create node
            source = Airport(row['SourceAirport'], row['SourceAirport_City'], row['SourceAirport_Country'],
                             row['SourceAirport_Latitude'], row['SourceAirport_Longitude'],
                             row['SourceAirport_Altitude'])
            destination = Airport(row['DestinationAirport'], row['DestinationAirport_City'],
                                  row['DestinationAirport_Country'],
                                  row['DestinationAirport_Latitude'], row['DestinationAirport_Longitude'],
                                  row['DestinationAirport_Altitude'])
            graph.add_nodes_from([source, destination])

            # create edge
            flight = Flight(row['Airline'], row['Distance'], row['FlyTime'], row['Price'])

            graph.add_weighted_edges_from([(source, destination, flight)])

        with open('graph.pkl', 'wb') as f:
            pickle.dump(graph, f)

    else:
        with open('graph.pkl', 'rb') as f:
            graph = pickle.load(f)

    source_inp, destination_inp = input().split(" - ")

    check_source, check_destination = False, False
    for vertex in graph.nodes:
        if check_source is True and check_destination is True:
            break
        if vertex.airport == source_inp:
            source_vertex = vertex
            check_source = True
        if vertex.airport == destination_inp:
            destination_vertex = vertex
            check_destination = True

    start_time = time.time()
    best_dijkstra = dijkstra(graph, source_vertex, destination_vertex)
    end_time = time.time()
    exetime_dijkstra = end_time - start_time

    start_time = time.time()
    best_astar = a_stare(graph, source_vertex, destination_vertex)
    end_time = time.time()
    exetime_astar = end_time - start_time

    #  write dijkstra

    with open('nine-UIAI4021-PR1-Q1(Dijkstra).txt', 'w', encoding='utf-8') as file_dijkstra:

        file_dijkstra.write("Dijkstra Algorithm\n")
        file_dijkstra.write("Execution Time: {}s\n".format(exetime_dijkstra))
        file_dijkstra.write(".-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-\n")

        total_duration = 0
        total_time = 0
        total_price = 0

        for i in range(len(best_dijkstra) - 1):
            file_dijkstra.write("Flight #{}".format(i + 1))
            flight_current = graph.get_edge_data(best_dijkstra[i], best_dijkstra[i + 1])['weight']
            file_dijkstra.write("({})\n".format(flight_current.airline))
            file_dijkstra.write("From: {airport} - {city}, {country}\n".format(
                airport=best_dijkstra[i].airport, city=best_dijkstra[i].city, country=best_dijkstra[i].country
            ))
            file_dijkstra.write("To: {airport} - {city}, {country}\n".format(
                airport=best_dijkstra[i + 1].airport, city=best_dijkstra[i + 1].city,
                country=best_dijkstra[i + 1].country
            ))
            file_dijkstra.write("Duration: {}km\n".format(flight_current.distance))
            total_duration += flight_current.distance
            file_dijkstra.write("Time: {}h\n".format(flight_current.fly_time))
            total_time += flight_current.fly_time
            file_dijkstra.write("Price: {}$\n".format(flight_current.price))
            total_price += flight_current.price
            file_dijkstra.write("----------------------------\n")

        file_dijkstra.write("Total Price: {}$\n".format(total_price))
        file_dijkstra.write("Total Duration: {}km\n".format(total_duration))
        file_dijkstra.write("Total Time: {}h\n".format(total_time))

    #  write astar

    with open('nine-UIAI4021-PR1-Q1(AStar).txt', 'w', encoding='utf-8') as file_astar:

        file_astar.write("A* Algorithm\n")
        file_astar.write("Execution Time: {}s\n".format(exetime_astar))
        file_astar.write(".-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-\n")

        total_duration = 0
        total_time = 0
        total_price = 0

        for i in range(len(best_astar) - 1):
            file_astar.write("Flight #{}".format(i + 1))
            flight_current = graph.get_edge_data(best_astar[i], best_astar[i + 1])['weight']
            file_astar.write("({})\n".format(flight_current.airline))
            file_astar.write("From: {airport} - {city}, {country}\n".format(
                airport=best_astar[i].airport, city=best_astar[i].city, country=best_astar[i].country
            ))
            file_astar.write("To: {airport} - {city}, {country}\n".format(
                airport=best_astar[i + 1].airport, city=best_astar[i + 1].city, country=best_astar[i + 1].country
            ))
            file_astar.write("Duration: {}km\n".format(flight_current.distance))
            total_duration += flight_current.distance
            file_astar.write("Time: {}h\n".format(flight_current.fly_time))
            total_time += flight_current.fly_time
            file_astar.write("Price: {}$\n".format(flight_current.price))
            total_price += flight_current.price
            file_astar.write("----------------------------\n")

        file_astar.write("Total Price: {}$\n".format(total_price))
        file_astar.write("Total Duration: {}km\n".format(total_duration))
        file_astar.write("Total Time: {}h\n".format(total_time))


