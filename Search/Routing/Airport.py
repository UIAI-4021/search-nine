# Airport --> Node
class Airport:
    def __init__(self, airport, city, country, latitude, longitude, altitude):
        self.airport = airport
        self.city = city
        self.country = country
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude

    def __eq__(self, other):
        return (self.airport == other.airport and self.city == other.city and self.country == other.country and
                self.latitude == other.latitude and self.longitude == other.longitude and self.altitude == other.altitude)

    def __hash__(self):
        return hash((self.airport, self.city, self.country, self.latitude, self.longitude, self.altitude))
