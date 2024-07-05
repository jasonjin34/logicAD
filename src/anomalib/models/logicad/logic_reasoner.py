from pyswip import Prolog
# Initialize Prolog
prolog = Prolog()

# Define the fact: one plate has two parts, left side contains apple, right side contains orange
prolog.assertz("plate_parts(plate([orange, orange, peach], [banana, nuts, granola]))")
query = list(prolog.query("plate_parts(plate(Left, _)), \\+ (Left = [orange])"))
print(query)  # Output: [{}] indicating the fact exists

query = list(prolog.query("plate_parts(plate([orange], [banana, nuts, granola]))"))
print(query)  # Output: [{}] indicating the fact exists

# # Function to check if a fact is true
# def check_fact(query):
#     result = list(prolog.query(query))
#     if result:
#         print(f"The fact '{query}' is true.")
#     else:
#         print(f"The fact '{query}' is false.")


# Define rules available items should have peach, tangarine or orange on the left side
# and granola, banana, nuts/almonds on the right side

# prolog.assertz("contains(left, Orange) :- member(Orange, [two_tangerine, two_orange, two_tangerines, two_oranges])")
# prolog.assertz("contains(left, peach)")
# prolog.assertz("leftcorrect(X, Y) :- contains(left, X), contains(left, Y)")
# prolog.assertz("contains(right, Banana) :- member(Banana, [banana, banana_chips])")
# prolog.assertz("contains(right, Nuts) :-  member(Nuts, [almonds, nuts])")
# prolog.assertz("contains(right, granola)")
# prolog.assertz("rightcorrect(X, Y, Z) :- contains(right, X), contains(right, Y), contains(right, Z)")
# 
# prolog.assertz("correct(X, Y, A, B, C) :- leftcorrect(X, Y), rightcorrect(A, B, C)")
# 
# objects = {
#   "left": {
#     "peach": 1,
#     "tangerines": 2
#   },
#   "right": {
#     "granola": 1,
#     "banana chips": 1,
#     "almonds": 1
#   }
# }
# 
# 
# check_fact("rightcorrect(nuts, banana, banana)")
# # check_fact("correct(peach, two_tangerines, banana_chips, almonds)")
# 
# # def convert_objects_to_prolog_query(objects):
# #     "recursive function to convert objects to prolog query"
# #     def number_to_string(number):
# #         if number == 1:
# #             return ""
# #         elif number >= 2:
# #             return f"{number}_"
# #         else:
# #             return "no_"
# #     def convert_to_prolog(objects, query=None):
# #         if query is None:
# #             query = []
# #         for key, value in objects.items():
# #             if isinstance(value, dict):
# #                 convert_to_prolog(value, query=query)
# #             else:
# #                 query.append(key)
# #         return query
# #     query = convert_to_prolog(objects)
# #     return query
# # 
# # query = convert_objects_to_prolog_query(objects["left"])
# # 
# # print(query)
# 
# 