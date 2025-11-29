from app import app

if __name__ == "__main__":
    with app.test_client() as c:
        resp = c.post('/shield_prompt', json={"prompt": "Hello, how are you?"})
        print("STATUS:", resp.status_code)
        print("JSON:")
        try:
            print(resp.get_json())
        except Exception as e:
            print("Parse error:", e)
            print("Raw:", resp.data[:200])
