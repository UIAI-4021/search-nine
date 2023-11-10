# Flight --> Edge
class Flight:
    def __init__(self, airline, distance, fly_time, price):
        self.airline = airline
        self.distance = distance
        self.fly_time = fly_time
        self.price = price

        self.cost_edge = price + fly_time + distance

    def __eq__(self, other):
        return (self.airline == other.airline and self.distance == other.distance and
                self.fly_time == other.fly_time and self.price == other.price)

    def __hash__(self):
        return hash((self.airline, self.distance, self.fly_time, self.price))
