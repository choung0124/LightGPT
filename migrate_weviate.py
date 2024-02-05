import weaviate
import weaviate.classes as wvc

client = weaviate.connect_to_local()  # Connect with default parameters

try:
    collection = client.collections.create(
        name="TestArticle",
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_cohere(),
        generative_config=wvc.config.Configure.Generative.cohere(),
        properties=[
            wvc.config.Property(
                name="title",
                data_type=wvc.config.DataType.TEXT
            )
        ]
    )

finally:
    client.close()