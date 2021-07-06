from django.db import models

# Create your models here.


class Prediction:
    date: str
    price: float

    def __init__(self, date, price):
        self.date = date
        self.price = price
