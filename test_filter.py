from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

metadata_filters = {}
tenant_id = 'aditya-ds'

conditions = []
if tenant_id:
    # Qdrant client might reject MatchValue without kwargs strictly or if type is weird
    conditions.append(FieldCondition(key='tenant_id', match=MatchValue(value=tenant_id)))

for field, cfg in metadata_filters.items():
    op = cfg.get('op')
    value = cfg.get('value')
    if value is None:
        continue

    if op == '$eq':
        conditions.append(FieldCondition(key=field, match=MatchValue(value=value)))
    elif op == '$in' and isinstance(value, list) and value:
        conditions.append(FieldCondition(key=field, match=MatchAny(any=value)))

if not conditions:
    print('No conditions')
else:
    f = Filter(must=conditions)
    print(f)
