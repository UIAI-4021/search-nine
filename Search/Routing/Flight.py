# Flight --> Edge
class Flight:
    def __init__(self,
                 airline, source_airport, destination_airport, source_airport_city, source_airport_country,
                 source_airport_latitude, source_airport_longitude, destination_airport_city,
                 destination_airport_country, destination_airport_latitude, destination_airport_longitude,
                 distance, fly_time, price):
        self.airline = airline
        self.source_airport = source_airport
        self.destination_airport = destination_airport
        self.source_airport_city = source_airport_city
        self.source_airport_country = source_airport_country
        self.source_airport_latitude = source_airport_latitude
        self.source_airport_longitude = source_airport_longitude
        self.destination_airport_city = destination_airport_city
        self.destination_airport_country = destination_airport_country
        self.destination_airport_latitude = destination_airport_latitude
        self.destination_airport_longitude = destination_airport_longitude
        self.distance = distance
        self.fly_time = fly_time
        self.price = price