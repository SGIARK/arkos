import psycopg2

ARK_DATABASE = "Inital_Database"
PG_PASSWORD = "your_password"

##Postgress createConnection and destroyConnection functions from: https://medium.com/@shrinandkadekodi/database-in-python-using-psycopg2-7c495cca1ab3


# # function to create connection
def createConnection(dbname):
    # this executes when no database has been created
    if dbname == ARK_DATABASE:
        con = psycopg2.connect(
            user="postgres", password=PG_PASSWORD, host="localhost", port="5432"
        )

    # this executes when database has been created
    else:
        con = psycopg2.connect(
            user="postgres",
            database=dbname,
            password=PG_PASSWORD,
            host="localhost",
            port="5432",
        )
    return con


# function to destroy connection
def destroyConnection(con):
    try:
        cursor = con.cursor()
        cursor.close()
        con.close()
        print("PostgreSQL connection is closed")
    except:
        pass


event_type = "CREATE TYPE Event AS (event_name TEXT, date DATE, state_at TIME WITH TIME ZONE, end_at TIME WITH TIME ZONE, reoccurance TEXT)"
task_type = "CREATE TYPE Task AS (task_name TEXT, date DATE, state_at TIME WITH TIME ZONE, end_at TIME WITH TIME ZONE, reoccurance TEXT, completion BOOLEAN)"

con = createConnection(ARK_DATABASE)
cursor = con.cursor()
cursor.execute(event_type)
cursor.execute(task_type)
con.commit()
destroyConnection(con)


# Can build entire calendar in postgres
# but not sure how we want to handle things
# https://medium.com/justdataplease/building-a-dynamic-date-calendar-in-postgresql-a-step-by-step-guide-20c8edfc3bf7
# database name and code to create table in the database


# ## CHROMA DB trial
# this might be an interesting way to store data
# mainly for the LLM part but postgres also has an embbedings implemention for future use
# postgress syntax is more of a learning curve so it might be easier to
# use chromabd for quicker trials.

""" 
import chromadb
chroma_client = chromadb.Client()

# switch `create_collection` to `get_or_create_collection` to avoid creating a new collection every time
collection = chroma_client.get_or_create_collection(name="my_collection")

# switch `add` to `upsert` to avoid adding the same documents every time
collection.upsert(
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges"
    ],
    ids=["id1", "id2"]
)

results = collection.query(
    query_texts=["This is a query document about florida"], # Chroma will embed this for you
    n_results=1 # how many results to return
)

print(results)

"""
