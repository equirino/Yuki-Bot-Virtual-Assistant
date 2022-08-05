import sqlite3
import json
from datetime import datetime

time_date = '2015-05'
start_row = 0
data_transfer = []

sqldb = sqlite3.connect('reddit_data/{}.db'.format(time_date))
mycursor = sqldb.cursor()


def sql_table():
    mycursor.execute(
        "CREATE TABLE IF NOT EXISTS parent_reply(parent_id TEXT PRIMARY KEY, "
        "comment_id TEXT UNIQUE, parent TEXT, comment TEXT, "
        "subreddit TEXT, unix INT, score INT)")


def format_data(data):
    data = data.replace('\n', ' ').replace('\r', ' ').replace('"', "'")
    return data


def acceptable(data):
    if len(data.split(' ')) > 50 or len(data) < 1 or len(data) > 1000 \
            or data == "[deleted]" or data == "[removed]":
        return False
    else:
        return True


def check_id(category, id_string, pid):
    try:
        match = "SELECT {} FROM parent_reply WHERE {} = '{}' LIMIT 1".format(category, id_string, pid)
        mycursor.execute(match)
        result = mycursor.fetchone()
        if result is not None:
            return result[0]
        else:
            return False
    except Exception as Error:
        print("FIND ID ERROR", str(Error))
        return False


def sql_commit(comment_data):
    data_transfer.append(comment_data)
    if len(data_transfer) > 1000:
        mycursor.execute('BEGIN TRANSACTION')
        for i in data_transfer:
            try:
                mycursor.execute(i)
            except:
                pass
        sqldb.commit()
        data_transfer.clear()


def sql_insert_replace_comment(commentid, parentid, parent, comment, subreddit, time, score_val):
    try:
        sql_update_query = """UPDATE parent_reply SET parent_id = ?, comment_id = ?, 
        parent = ?, comment = ?, subreddit = ?, unix = ?, score = ? WHERE parent_id = ?;""".format(
            parentid, commentid, parent, comment, subreddit, int(time), score_val, parentid)
        sql_commit(sql_update_query)
    except Exception as Error:
        print("INSERT REPLACE ERROR", str(Error))


def sql_insert_has_parent(parentid, commentid, parent, comment, subreddit_name, time, score_val):
    try:
        sql_insert_query = """INSERT INTO parent_reply (parent_id, comment_id, parent, comment, subreddit, unix, score)
        VALUES ("{}","{}","{}","{}","{}",{},{});""".format(parentid, commentid, parent, comment, subreddit_name,
                                                           int(time), score_val)
        sql_commit(sql_insert_query)
    except Exception as Error:
        print("INSERT PARENT ERROR", str(Error))


def sql_insert_with_no_parent(parentid, commentid, comment, subreddit_name, time, score_val):
    try:
        sql_insert_query = """INSERT INTO parent_reply (parent_id, comment_id, comment, subreddit, unix, score)
        VALUES ("{}","{}","{}","{}",{},{});""".format(parentid, commentid, comment, subreddit_name,
                                                      int(time), score_val)
        sql_commit(sql_insert_query)
    except Exception as Error:
        print("INSERT NO PARENT ERROR", str(Error))


def main():
    sql_table()
    row_counter = 0
    paired_rows = 0

    with open('C:/Users/equir/PycharmProjects/PersonalProjects/chatbot/data/reddit_data/RC_{}'.format(time_date),
              buffering=1000) as file:
        for comment_sect in file:
            row_counter += 1

            if row_counter > start_row:
                try:
                    comment_sect = json.loads(comment_sect)
                    parent_id = comment_sect['parent_id']
                    body = format_data(comment_sect['body'])
                    created_utc = comment_sect['created_utc']
                    score = comment_sect['score']
                    comment_id = comment_sect['name']
                    subreddit = comment_sect['subreddit']
                    parent_data = check_id("comment", "comment_id", parent_id)

                    if score >= 2:
                        existing_comment_score = check_id("score", "parent_id", parent_id)

                        if existing_comment_score and score > existing_comment_score and acceptable(body):
                            sql_insert_replace_comment(parent_id, comment_id, parent_data, body, subreddit, created_utc,
                                                       score)
                        else:
                            if acceptable(body) and parent_data:
                                sql_insert_has_parent(parent_id, comment_id, parent_data, body, subreddit, created_utc,
                                                      score)
                                paired_rows += 1
                            else:
                                sql_insert_with_no_parent(parent_id, comment_id, body, subreddit, created_utc, score)
                except Exception as Error:
                    print(str(Error))

            if row_counter % 100000 == 0:
                print('Total Rows Read: {}, Paired Rows: {}, Time: {}'.format(row_counter, paired_rows,
                                                                              str(datetime.now())))

            if row_counter > start_row and row_counter % 1000000 == 0:
                print("Cleaning database")
                sql_clean_query = "DELETE FROM parent_reply WHERE parent is NULL"
                mycursor.execute(sql_clean_query)
                sqldb.commit()
                mycursor.execute("VACUUM")
                sqldb.commit()

            if row_counter % 3250000 == 0:
                print("Ending database creation process")
                break


if __name__ == '__main__':
    main()
