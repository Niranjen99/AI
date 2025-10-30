from memory import vectorstore

results = vectorstore.similarity_search("James", k=3)
'''results = vectorstore.similarity_search(
    "Max",
    k=5,
    filter={"user_id": "user_123"}
)'''

for r in results:
    print(r.page_content)
