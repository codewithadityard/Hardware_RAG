import redis


class RedisCahce:
    def __init__(self,host='localhost',port=6379,db=0,ttl=86400):
        self.ttl=ttl

        try:
            self.client=redis.Redis(host=host,port=port,db=db,decode_responses=True)
            self.client.ping()
            print("[System] Successfully connected to Redis Cache Database.")

        except redis.ConnectionError:
            print("[System Error] CRITICAL: Could not connect to Redis on port 6379.")
            print("Ensure the Redis server is installed and running in the background.")
            self.client = None

    def check_cache(self, query: str):

            if not self.client:
                return None
            
            normalized_query = query.strip().lower()
            return self.client.get(normalized_query)
        

    def save_to_cache(self, query: str, answer: str):
            if not self.client:
                return
                
            normalized_query = query.strip().lower()
            # ex=self.ttl tells Redis to automatically delete this record after 24 hours
            self.client.set(name=normalized_query, value=answer, ex=self.ttl)

    def get_len(self):
        if not self.client:
            return 0
        return self.client.dbsize()
    
    