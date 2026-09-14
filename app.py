import os
import sys
import io
import math
import json
import base64
import socket
import webbrowser
import threading
from itertools import permutations
from flask import Flask, render_template, request, jsonify
import re
import requests
import qrcode

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

# Endpoints
OSRM_TABLE_URL = "https://router.project-osrm.org/table/v1/driving/{coords}?annotations=distance,duration"
OSRM_ROUTE_URL = "https://router.project-osrm.org/route/v1/driving/{coords}?overview=full&geometries=geojson&steps=true"
NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
NP_API_URL = "https://api.novaposhta.ua/v2.0/json/"

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000.0  # meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0) ** 2 +         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def get_distance_duration_matrix(points):
    """
    Retrieves real road distance (meters) and duration (seconds) matrix from OSRM.
    Falls back to Haversine if offline or request fails.
    """
    n = len(points)
    coords_str = ";".join([f"{p['lng']:.6f},{p['lat']:.6f}" for p in points])
    url = OSRM_TABLE_URL.format(coords=coords_str)
    
    try:
        resp = requests.get(url, timeout=7, headers={"User-Agent": "TransportTSPApp/1.0"})
        if resp.status_code == 200:
            data = resp.json()
            if data.get("code") == "Ok":
                dist_matrix = data.get("distances", [])
                dur_matrix = data.get("durations", [])
                if dist_matrix and dur_matrix:
                    for i in range(n):
                        for j in range(n):
                            if dist_matrix[i][j] is None:
                                dist_matrix[i][j] = haversine_distance(points[i]['lat'], points[i]['lng'], points[j]['lat'], points[j]['lng']) * 1.35
                            if dur_matrix[i][j] is None:
                                dur_matrix[i][j] = dist_matrix[i][j] / 13.88
                    return dist_matrix, dur_matrix, True
    except Exception as e:
        print(f"OSRM Table error: {e}", file=sys.stderr)

    dist_matrix = [[0.0] * n for _ in range(n)]
    dur_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                d = haversine_distance(points[i]['lat'], points[i]['lng'], points[j]['lat'], points[j]['lng']) * 1.35
                dist_matrix[i][j] = d
                dur_matrix[i][j] = d / 15.0 # ~54 km/h
    return dist_matrix, dur_matrix, False

def solve_tsp_exact(cost_matrix, mode="round_trip", fixed_end=None):
    n = len(cost_matrix)
    if n <= 1:
        return [0]
    if n == 2:
        return [0, 1] if mode != "round_trip" else [0, 1, 0]

    start = 0
    if mode == "fixed_end":
        end = fixed_end if fixed_end is not None else n - 1
        intermediate = [i for i in range(n) if i != start and i != end]
        best_cost = float('inf')
        best_path = [start] + intermediate + [end]
        for perm in permutations(intermediate):
            path = [start] + list(perm) + [end]
            cost = sum(cost_matrix[path[k]][path[k+1]] for k in range(len(path)-1))
            if cost < best_cost:
                best_cost = cost
                best_path = path
        return best_path

    elif mode == "any_end":
        other_nodes = [i for i in range(1, n)]
        best_cost = float('inf')
        best_path = [start] + other_nodes
        for perm in permutations(other_nodes):
            path = [start] + list(perm)
            cost = sum(cost_matrix[path[k]][path[k+1]] for k in range(len(path)-1))
            if cost < best_cost:
                best_cost = cost
                best_path = path
        return best_path

    else: # round_trip
        other_nodes = [i for i in range(1, n)]
        best_cost = float('inf')
        best_path = [start] + other_nodes + [start]
        for perm in permutations(other_nodes):
            path = [start] + list(perm) + [start]
            cost = sum(cost_matrix[path[k]][path[k+1]] for k in range(len(path)-1))
            if cost < best_cost:
                best_cost = cost
                best_path = path
        return best_path

def solve_tsp_heuristic(cost_matrix, mode="round_trip", fixed_end=None):
    n = len(cost_matrix)
    start = 0

    unvisited = set(range(n))
    unvisited.remove(start)
    if mode == "fixed_end":
        end = fixed_end if fixed_end is not None else n - 1
        if end in unvisited:
            unvisited.remove(end)
    else:
        end = None

    current = start
    path = [start]
    while unvisited:
        next_node = min(unvisited, key=lambda x: cost_matrix[current][x])
        path.append(next_node)
        unvisited.remove(next_node)
        current = next_node

    if mode == "fixed_end":
        path.append(end)
    elif mode == "round_trip":
        path.append(start)

    def path_cost(p):
        return sum(cost_matrix[p[i]][p[i+1]] for i in range(len(p)-1))

    improved = True
    iteration = 0
    max_idx = len(path) - (1 if mode in ["fixed_end", "round_trip"] else 0)
    while improved and iteration < 200:
        improved = False
        iteration += 1
        for i in range(1, max_idx - 1):
            for j in range(i + 1, max_idx):
                new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                if path_cost(new_path) < path_cost(path) - 1e-6:
                    path = new_path
                    improved = True
                    break
            if improved:
                break

    return path

def solve_tsp(cost_matrix, mode="round_trip", fixed_end=None):
    n = len(cost_matrix)
    if n <= 10:
        return solve_tsp_exact(cost_matrix, mode, fixed_end)
    else:
        return solve_tsp_heuristic(cost_matrix, mode, fixed_end)

def get_route_geometry(ordered_points):
    coords_str = ";".join([f"{p['lng']:.6f},{p['lat']:.6f}" for p in ordered_points])
    url = OSRM_ROUTE_URL.format(coords=coords_str)
    
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "TransportTSPApp/1.0"})
        if resp.status_code == 200:
            data = resp.json()
            if data.get("code") == "Ok" and data.get("routes"):
                route = data["routes"][0]
                geometry = route.get("geometry")
                legs = route.get("legs", [])
                leg_details = []
                for leg in legs:
                    leg_coords = []
                    for step in leg.get("steps", []):
                        step_coords = step.get("geometry", {}).get("coordinates", [])
                        if step_coords:
                            if leg_coords and leg_coords[-1] == step_coords[0]:
                                leg_coords.extend(step_coords[1:])
                            else:
                                leg_coords.extend(step_coords)
                    
                    leg_details.append({
                        "distance_km": round(leg.get("distance", 0) / 1000.0, 2),
                        "duration_min": round(leg.get("duration", 0) / 60.0, 1),
                        "summary": leg.get("summary", ""),
                        "coordinates": leg_coords
                    })
                return {
                    "geometry": geometry,
                    "legs": leg_details,
                    "distance_km": round(route.get("distance", 0) / 1000.0, 2),
                    "duration_min": round(route.get("duration", 0) / 60.0, 1)
                }
    except Exception as e:
        print(f"OSRM Route error: {e}", file=sys.stderr)

    coordinates = [[p['lng'], p['lat']] for p in ordered_points]
    total_dist = 0.0
    legs = []
    for i in range(len(ordered_points) - 1):
        p1, p2 = ordered_points[i], ordered_points[i+1]
        d = haversine_distance(p1['lat'], p1['lng'], p2['lat'], p2['lng']) * 1.35
        total_dist += d
        legs.append({
            "distance_km": round(d / 1000.0, 2),
            "duration_min": round((d / 15.0) / 60.0, 1),
            "summary": "Маршрут по прямій (офлайн)",
            "coordinates": [[p1['lng'], p1['lat']], [p2['lng'], p2['lat']]]
        })
    return {
        "geometry": {
            "type": "LineString",
            "coordinates": coordinates
        },
        "legs": legs,
        "distance_km": round(total_dist / 1000.0, 2),
        "duration_min": round((total_dist / 15.0) / 60.0, 1)
    }

def generate_google_maps_url(ordered_points):
    coords_list = [f"{p['lat']:.6f},{p['lng']:.6f}" for p in ordered_points]
    return "https://www.google.com/maps/dir/" + "/".join(coords_list)

def generate_qr_base64(url):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=8,
        border=3,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#1a73e8", back_color="#ffffff")
    
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

@app.route("/")
def index():
    return render_template("index.html")

def is_nova_poshta_query(q):
    return bool(re.search(r'(?i)\b(нова\s*пошта|нової\s*пошти|нових\s*пошт|новими\s*поштами|нові\s*пошти|нп|novaposhta|nova\s*poshta)\b', q))

def clean_np_query(q):
    text = re.sub(r'(?i)\b(нова\s*пошта|нової\s*пошти|нових\s*пошт|новими\s*поштами|нові\s*пошти|нп|novaposhta|nova\s*poshta)\b', '', q)
    wh_match = re.search(r'(?:№|відділення|поштомат|склад)?\s*(\d+)', text)
    wh_number = wh_match.group(1) if wh_match else None
    
    settlement = re.sub(r'(?i)\b(відділення|поштомат|пункт|село|смт|місто|м\.|с\.|вулиця|вул\.)\b', '', text)
    if wh_number:
        settlement = settlement.replace(wh_number, '')
    settlement = re.sub(r'[№#,.]', '', settlement).strip()
    return settlement, wh_number

def search_nova_poshta(query, lat=None, lng=None):
    settlement, wh_num = clean_np_query(query)
    results = []

    # If no settlement in query, reverse geocode map center to get current city/town
    if not settlement and lat is not None and lng is not None:
        try:
            r = requests.get(NOMINATIM_REVERSE_URL, params={'lat': lat, 'lon': lng, 'format': 'json'}, headers={'User-Agent': 'TransportTSP/1.0'}, timeout=4).json()
            addr = r.get('address', {})
            settlement = addr.get('city') or addr.get('town') or addr.get('village') or addr.get('suburb') or addr.get('county')
        except Exception as e:
            print(f"Reverse geocode error: {e}", file=sys.stderr)

    if not settlement:
        settlement = "Київ"

    try:
        p_settle = {
            "modelName": "AddressGeneral",
            "calledMethod": "searchSettlements",
            "methodProperties": {"CityName": settlement, "Limit": "4"}
        }
        resp = requests.post(NP_API_URL, json=p_settle, timeout=5).json()
        data = resp.get("data", [])
        if data:
            addresses = data[0].get("Addresses", [])
            for addr in addresses[:2]:
                s_ref = addr.get("Ref")
                s_present = addr.get("Present")
                
                p_wh = {
                    "modelName": "AddressGeneral",
                    "calledMethod": "getWarehouses",
                    "methodProperties": {
                        "SettlementRef": s_ref,
                        "Limit": "40"
                    }
                }
                if wh_num:
                    p_wh["methodProperties"]["WarehouseId"] = wh_num

                resp_wh = requests.post(NP_API_URL, json=p_wh, timeout=5).json()
                whs = resp_wh.get("data", [])
                for w in whs:
                    w_lat = w.get("Latitude")
                    w_lon = w.get("Longitude")
                    if w_lat and w_lon and float(w_lat) > 0:
                        name = f"Нова Пошта: {w.get('Description')}"
                        results.append({
                            "name": name,
                            "short_name": w.get("Description"),
                            "address": f"{s_present}, {w.get('ShortAddress')}",
                            "lat": float(w_lat),
                            "lng": float(w_lon),
                            "category": "novaposhta",
                            "icon": "📦"
                        })
    except Exception as e:
        print(f"NP search error: {e}", file=sys.stderr)

    return results

def format_nominatim_item(item, query):
    raw_name = item.get("name") or item.get("display_name", "")
    display = item.get("display_name", "")
    parts = [p.strip() for p in display.split(",") if p.strip()]
    
    category = item.get("type", "")
    icon = "📍"
    q_low = query.lower()
    if "атб" in q_low:
        icon = "🛒"
    elif "сільпо" in q_low or "silpo" in q_low:
        icon = "🛒"
    elif "пошта" in q_low or "post" in category:
        icon = "📦"
    elif "окко" in q_low or "wog" in q_low or "fuel" in category:
        icon = "⛽"
    elif "аптека" in q_low or "pharmacy" in category:
        icon = "💊"
    elif "лікарня" in q_low or "hospital" in category:
        icon = "🏥"

    title = parts[0] if parts else raw_name
    address = ", ".join(parts[1:4]) if len(parts) > 1 else display

    return {
        "name": f"{title} ({address})",
        "short_name": title,
        "address": address,
        "lat": float(item["lat"]),
        "lng": float(item["lon"]),
        "icon": icon,
        "category": category
    }

@app.route("/api/gas_stations")
def get_gas_stations():
    lat = request.args.get("lat", type=float)
    lng = request.args.get("lng", type=float)
    if lat is None or lng is None:
        return jsonify([])
    
    delta = 0.09
    viewbox = f"{lng-delta:.5f},{lat+delta:.5f},{lng+delta:.5f},{lat-delta:.5f}"
    headers = {"User-Agent": "TransportTSPApp/1.0"}
    
    results = []
    try:
        resp = requests.get(
            NOMINATIM_SEARCH_URL,
            params={"q": "АЗС", "format": "json", "limit": 10, "addressdetails": 1, "viewbox": viewbox, "bounded": 1},
            headers=headers,
            timeout=5
        )
        if resp.status_code == 200:
            for item in resp.json():
                results.append(format_nominatim_item(item, "АЗС"))
    except Exception as e:
        print(f"Gas stations error: {e}", file=sys.stderr)
        
    return jsonify(results)

@app.route("/api/search")
def search_address():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])
    
    lat = request.args.get("lat", type=float)
    lng = request.args.get("lng", type=float)
    viewbox = request.args.get("viewbox", "").strip()
    
    # 1. Nova Poshta query
    if is_nova_poshta_query(q):
        np_results = search_nova_poshta(q, lat, lng)
        if np_results:
            return jsonify(np_results)
    
    results = []
    headers = {"User-Agent": "TransportOptimizationApp/1.0"}

    # 2. Nominatim with map viewbox
    try:
        if viewbox:
            resp = requests.get(
                NOMINATIM_SEARCH_URL,
                params={"q": q, "format": "json", "limit": 20, "addressdetails": 1, "viewbox": viewbox, "bounded": 1},
                headers=headers,
                timeout=5
            )
            if resp.status_code == 200:
                for item in resp.json():
                    results.append(format_nominatim_item(item, q))

        # If bounded search found few or no items, search across Ukraine
        if len(results) < 3:
            resp = requests.get(
                NOMINATIM_SEARCH_URL,
                params={"q": q, "format": "json", "limit": 15, "addressdetails": 1, "countrycodes": "ua"},
                headers=headers,
                timeout=5
            )
            if resp.status_code == 200:
                existing_coords = {(round(r["lat"], 4), round(r["lng"], 4)) for r in results}
                for item in resp.json():
                    lat_val = float(item["lat"])
                    lng_val = float(item["lon"])
                    if (round(lat_val, 4), round(lng_val, 4)) not in existing_coords:
                        results.append(format_nominatim_item(item, q))
        
        # If still empty and query has "пошта", try Nova Poshta fallback
        if not results and ("пошта" in q.lower() or "почта" in q.lower() or "нп" in q.lower()):
            np_res = search_nova_poshta(q, lat, lng)
            if np_res:
                return jsonify(np_res)

    except Exception as e:
        print(f"Search error: {e}", file=sys.stderr)

    return jsonify(results)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2.0)**2
    return 2.0 * R * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

def find_coord_at_km(coords, target_km):
    if not coords or target_km <= 0:
        return {"lat": coords[0][1], "lng": coords[0][0]} if coords else {"lat": 0, "lng": 0}
    accum = 0.0
    for i in range(len(coords) - 1):
        p1 = coords[i]
        p2 = coords[i+1]
        seg_dist = haversine_km(p1[1], p1[0], p2[1], p2[0])
        if accum + seg_dist >= target_km:
            t = (target_km - accum) / seg_dist if seg_dist > 0 else 0
            lat = p1[1] + t * (p2[1] - p1[1])
            lng = p1[0] + t * (p2[0] - p1[0])
            return {"lat": round(lat, 6), "lng": round(lng, 6)}
        accum += seg_dist
    return {"lat": coords[-1][1], "lng": coords[-1][0]}

def simulate_refueling(ordered_points, route_data, fuel_consumption, fuel_price, tank_capacity, start_fuel, reserve_fuel):
    coords = route_data.get("geometry", {}).get("coordinates", [])
    legs = route_data.get("legs", [])
    
    if fuel_consumption <= 0: fuel_consumption = 8.0
    if tank_capacity <= 0: tank_capacity = 50.0
    if start_fuel <= 0: start_fuel = tank_capacity * 0.5
    if reserve_fuel <= 0: reserve_fuel = tank_capacity * 0.15

    current_fuel = min(start_fuel, tank_capacity)
    refuel_events = []
    accum_km = 0.0
    total_refuel_liters = 0.0
    total_refuel_cost = 0.0

    for leg_idx, leg in enumerate(legs):
        leg_km = leg.get("distance_km", 0.0)
        leg_fuel = leg_km * (fuel_consumption / 100.0)

        from_name = ordered_points[leg_idx].get("name", f"Точка {leg_idx+1}") if leg_idx < len(ordered_points) else ""
        to_name = ordered_points[leg_idx+1].get("name", f"Точка {leg_idx+2}") if (leg_idx+1) < len(ordered_points) else ""

        # Check if fuel drops below reserve
        if current_fuel - leg_fuel < reserve_fuel:
            # Need to refuel at current stop before departing
            refuel_amount = round(tank_capacity - current_fuel, 1)
            if refuel_amount > 2.0:
                cost = round(refuel_amount * fuel_price, 2)
                total_refuel_liters += refuel_amount
                total_refuel_cost += cost
                stop_coord = {"lat": ordered_points[leg_idx]["lat"], "lng": ordered_points[leg_idx]["lng"]}
                refuel_events.append({
                    "at_km": round(accum_km, 1),
                    "location_type": "stop",
                    "stop_index": leg_idx,
                    "description": f"Дозаправка на зупинці '{from_name}' перед виїздом до '{to_name}'",
                    "fuel_before": round(current_fuel, 1),
                    "refuel_liters": refuel_amount,
                    "refuel_cost_uah": cost,
                    "lat": stop_coord["lat"],
                    "lng": stop_coord["lng"]
                })
                current_fuel = tank_capacity

            # If leg is longer than an entire full tank
            while current_fuel - leg_fuel < reserve_fuel and (tank_capacity - reserve_fuel) > 0:
                usable_km = (current_fuel - reserve_fuel) / (fuel_consumption / 100.0)
                refuel_km = accum_km + usable_km
                mid_coord = find_coord_at_km(coords, refuel_km)
                refuel_amount = round(tank_capacity - reserve_fuel, 1)
                cost = round(refuel_amount * fuel_price, 2)
                total_refuel_liters += refuel_amount
                total_refuel_cost += cost

                refuel_events.append({
                    "at_km": round(refuel_km, 1),
                    "location_type": "highway",
                    "stop_index": leg_idx,
                    "description": f"Дозаправка на АЗС на трасі (~{round(refuel_km, 1)} км)",
                    "fuel_before": round(reserve_fuel, 1),
                    "refuel_liters": refuel_amount,
                    "refuel_cost_uah": cost,
                    "lat": mid_coord["lat"],
                    "lng": mid_coord["lng"]
                })
                current_fuel = tank_capacity
                leg_km -= usable_km
                leg_fuel = leg_km * (fuel_consumption / 100.0)

        current_fuel = max(0.0, current_fuel - leg_fuel)
        accum_km += leg.get("distance_km", 0.0)

    return {
        "needs_refuel": len(refuel_events) > 0,
        "refuel_count": len(refuel_events),
        "events": refuel_events,
        "total_refuel_liters": round(total_refuel_liters, 1),
        "total_refuel_cost_uah": round(total_refuel_cost, 2),
        "final_fuel_liters": round(current_fuel, 1),
        "start_fuel_liters": round(start_fuel, 1),
        "tank_capacity": round(tank_capacity, 1),
        "reserve_fuel": round(reserve_fuel, 1)
    }

@app.route("/api/optimize", methods=["POST"])
def optimize_route():
    data = request.get_json() or {}
    points = data.get("points", [])
    vehicle_type = data.get("vehicle_type", "car")
    fuel_consumption = float(data.get("fuel_consumption", 7.5))
    fuel_price = float(data.get("fuel_price", 56.0))
    tank_capacity = float(data.get("tank_capacity", 50.0))
    start_fuel = float(data.get("start_fuel", tank_capacity * 0.5))
    reserve_fuel = float(data.get("reserve_fuel", tank_capacity * 0.15))
    mode = data.get("mode", "round_trip")
    objective = data.get("objective", "distance")

    if len(points) < 2:
        return jsonify({"error": "Необхідно обрати щонайменше 2 точки на карті"}), 400

    dist_matrix, dur_matrix, is_osrm = get_distance_duration_matrix(points)

    if objective == "time":
        cost_matrix = dur_matrix
    elif objective == "fuel":
        cost_matrix = dist_matrix
    else:
        cost_matrix = dist_matrix

    if mode == "custom":
        optimal_indices = list(range(len(points)))
    else:
        fixed_end = len(points) - 1 if mode == "fixed_end" else None
        optimal_indices = solve_tsp(cost_matrix, mode=mode, fixed_end=fixed_end)
    ordered_points = [points[i] for i in optimal_indices]

    original_indices = list(range(len(points)))
    if mode == "round_trip":
        original_indices.append(0)
    orig_dist_m = sum(dist_matrix[original_indices[k]][original_indices[k+1]] for k in range(len(original_indices)-1))
    orig_dur_s = sum(dur_matrix[original_indices[k]][original_indices[k+1]] for k in range(len(original_indices)-1))
    orig_dist_km = round(orig_dist_m / 1000.0, 2)
    orig_dur_min = round(orig_dur_s / 60.0, 1)

    route_data = get_route_geometry(ordered_points)

    total_km = route_data["distance_km"]
    total_min = route_data["duration_min"]
    hours = int(total_min // 60)
    mins = int(round(total_min % 60))
    if mins == 60:
        hours += 1
        mins = 0
    time_formatted = f"{hours} год {mins} хв" if hours > 0 else f"{mins} хв"

    fuel_liters = round((total_km * fuel_consumption) / 100.0, 2)
    total_cost_uah = round(fuel_liters * fuel_price, 2)

    saved_km = round(max(0.0, orig_dist_km - total_km), 2)
    saved_fuel = round((saved_km * fuel_consumption) / 100.0, 2)
    saved_uah = round(saved_fuel * fuel_price, 2)
    saved_min = round(max(0.0, orig_dur_min - total_min), 1)

    # Refueling simulation
    refueling_info = simulate_refueling(
        ordered_points, route_data, fuel_consumption, fuel_price, tank_capacity, start_fuel, reserve_fuel
    )

    gmaps_url = generate_google_maps_url(ordered_points)
    qr_data_url = generate_qr_base64(gmaps_url)

    stops = []
    for step_num, (idx, pt) in enumerate(zip(optimal_indices, ordered_points)):
        is_first = (step_num == 0)
        is_last = (step_num == len(ordered_points) - 1)
        role = "Старт" if is_first else ("Фініш (Повернення)" if (is_last and mode == "round_trip") else ("Фініш" if is_last else f"Зупинка {step_num}"))
        leg_info = route_data["legs"][step_num] if step_num < len(route_data["legs"]) else None
        
        stops.append({
            "step": step_num + 1,
            "original_index": idx,
            "name": pt.get("name", f"Точка {idx + 1}"),
            "lat": pt["lat"],
            "lng": pt["lng"],
            "role": role,
            "leg_to_next": leg_info
        })

    return jsonify({
        "success": True,
        "is_osrm": is_osrm,
        "vehicle_type": vehicle_type,
        "optimal_order": optimal_indices,
        "stops": stops,
        "route_geometry": route_data["geometry"],
        "metrics": {
            "total_km": total_km,
            "total_min": total_min,
            "time_formatted": time_formatted,
            "fuel_liters": fuel_liters,
            "total_cost_uah": total_cost_uah,
            "saved_km": saved_km,
            "saved_fuel": saved_fuel,
            "saved_uah": saved_uah,
            "saved_min": saved_min
        },
        "refueling": refueling_info,
        "google_maps_url": gmaps_url,
        "qr_code": qr_data_url
    })

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

@app.route("/api/network_info")
def network_info():
    ip = get_local_ip()
    port = request.host.split(":")[-1] if ":" in request.host else 5000
    mobile_url = f"http://{ip}:{port}"
    qr = generate_qr_base64(mobile_url)
    return jsonify({
        "local_ip": ip,
        "port": port,
        "mobile_url": mobile_url,
        "qr_code": qr
    })

def find_free_port(start_port=5000):
    for p in range(start_port, start_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', p)) != 0:
                return p
    return start_port

def open_browser(port):
    webbrowser.open(f"http://127.0.0.1:{port}")

if __name__ == "__main__":
    port = find_free_port(5000)
    local_ip = get_local_ip()
    mobile_url = f"http://{local_ip}:{port}"
    threading.Timer(1.2, lambda: open_browser(port)).start()
    print("\n" + "=" * 65)
    print("  🚗 ПРОГРАМА ОПТИМІЗАЦІЇ МАРШРУТУ УСПІШНО ЗАПУЩЕНА!  ")
    print(f"  💻 На комп'ютері:  http://127.0.0.1:{port}")
    print(f"  📱 НА ТЕЛЕФОНІ:    {mobile_url}")
    print("=" * 65)
    print("  💡 Для відкриття на телефоні: підключіть телефон до того ж Wi-Fi")
    print(f"     та введіть у браузері {mobile_url} або відскануйте QR в додатку!\n")
    app.run(host="0.0.0.0", port=port, debug=False)
