# Airport --> Node
class Airport:
    def __init__(self, airport, city, country):
        self.airport = airport
        self.city = city
        self.country = country

    def __eq__(self, other):
        return self.airport == other.airport and self.city == other.city and self.country == other.country

    def __hash__(self):
        return hash((self.airport, self.city, self.country))
