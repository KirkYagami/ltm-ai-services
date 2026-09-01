#write your code here

from pymongo import MongoClient

# Connect to MongoDB
client=MongoClient("mongodb://127.0.0.1:27017/")
db = client["hotel_booking"]

rooms = db["rooms"]
hotels = db["hotels"]



def find_rooms(room_type):
    pipeline = [
        {
            '$match':
            {
                "room_type":room_type
            } 
        },

        {

            "$lookup": {
                "from": "hotels",
                "localField": "hotel_id",
                "foreignField": "_id",
                "as": "hotel"
            }
        },

        {

            "$unwind": {
                "path": "$hotel",
                "preserveNullAndEmptyArrays": True
            }
        },

        {
            "$project":
            {
                "_id": 0,
                "room_number": 1,
                "roomType": 1,
                "price_per_night": 1,
                "hotel_name": {
                    "ifNull": [
                        "$hotel.hotel_name",
                        "Independent Room"
                    ]
                },

                               "location": {
                    "$ifNull": [
                        "$hotel.location",
                        "N/A"
                    ]
                }
            }
        }


    ]


    return list(rooms.aggregate(pipeline))



if __name__ == "__main__":
    room_type = input("Enter room type (Deluxe, Suite, Standard): ")
    available_rooms = find_rooms(room_type)

    if available_rooms:        
        print(f"\nAvailable {room_type} Rooms:")
        for room in available_rooms:
            print(f"Room Number: {room['room_number']}, Price: ${room['price_per_night']}, "
                  f"Hotel: {room['hotel_name']}, Location: {room['location']}")
    else:
        print(f"\nNo {room_type} rooms found.")