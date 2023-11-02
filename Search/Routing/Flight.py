# Flight --> Edge
class Flight:
    def __init__(self,
                 airline, source_airport, destination_airport, source_airport_city, source_airport_country,
                 source_airport_latitude, source_airport_longitude, source_airport_altitude, destination_airport_city,
                 destination_airport_country, destination_airport_latitude, destination_airport_longitude,
                 destination_airport_altitude, distance, fly_time, price):
        self.airline = airline
        self.source_airport = source_airport
        self.destination_airport = destination_airport
        self.source_airport_city = source_airport_city
        self.source_airport_country = source_airport_country
        self.source_airport_latitude = source_airport_latitude
        self.source_airport_longitude = source_airport_longitude
        self.source_airport_altitude = source_airport_altitude
        self.destination_airport_city = destination_airport_city
        self.destination_airport_country = destination_airport_country
        self.destination_airport_latitude = destination_airport_latitude
        self.destination_airport_longitude = destination_airport_longitude
        self.destination_airport_altitude = destination_airport_altitude
        self.distance = distance
        self.fly_time = fly_time
        self.price = price

        self.cost_edge =  price + fly_time + distance

    def __eq__(self, other):
        return (self.airline == other.airline and self.source_airport == other.source_airport and
                self.destination_airport == other.destination_airport and
                self.source_airport_city == other.source_airport_city and
                self.source_airport_country == other.source_airport_country and
                self.source_airport_latitude == other.source_airport_latitude and
                self.source_airport_longitude == other.source_airport_longitude and
                self.source_airport_altitude == other.source_airport_altitude and
                self.destination_airport_city == other.destination_airport_city and
                self.destination_airport_country == other.destination_airport_country and
                self.destination_airport_latitude == other.destination_airport_latitude and
                self.destination_airport_longitude == other.destination_airport_longitude and
                self.destination_airport_altitude == other.destination_airport_altitude and
                self.distance == other.distance and self.fly_time == other.fly_time and self.price == other.price)

    def __hash__(self):
        return hash((self.airline, self.source_airport, self.destination_airport,
                     self.source_airport_city, self.source_airport_country, self.source_airport_latitude,
                     self.source_airport_longitude, self.source_airport_altitude, self.destination_airport_city,
                     self.destination_airport_country, self.destination_airport_latitude,
                     self.destination_airport_longitude, self.destination_airport_altitude,
                     self.distance, self.fly_time, self.price))
