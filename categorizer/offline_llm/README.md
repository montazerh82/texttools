in this module we will use the batch API of openai

to process the data in a less expensive way

```python
off = OfflineCategorizer(
    client=client,
    categories=MyCategoryEnum,
    model="gpt-4o-2024-08-06",
    handlers=[FileResultHandler("out.csv")],
)

# Kick off the batch jobs (state is saved to .offline_categorizer_state.json)
off.start_process(my_long_list_of_texts)

# Later, or even after a restart, poll until results appear
while True:
    res = off.check_status()
    if res is not None:
        break
    time.sleep(30)

print("Done:", res)

```