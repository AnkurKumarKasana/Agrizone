from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from .models import ChatMessage
import json
import datetime
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def chat_home(request):
    recent_chats = ChatMessage.objects.all()[:10]
    return render(request, 'chatbot/chat.html', {'recent_chats': recent_chats})

@csrf_exempt
def send_message(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_message = data.get('message', '').strip()
            
            if not user_message:
                return JsonResponse({'success': False, 'error': 'Message cannot be empty'})
            
            # Get AI response
            bot_response = get_ai_response(user_message)
            
            # Save to database
            try:
                chat = ChatMessage.objects.create(
                    user_message=user_message,
                    bot_response=bot_response
                )
                timestamp = chat.created_at.strftime('%H:%M')
            except:
                timestamp = datetime.datetime.now().strftime('%H:%M')
            
            return JsonResponse({
                'success': True,
                'response': bot_response,
                'timestamp': timestamp
            })
            
        except Exception as e:
            print(f"Error in send_message: {e}")
            return JsonResponse({
                'success': True,
                'response': get_fallback_response(user_message),
                'timestamp': datetime.datetime.now().strftime('%H:%M')
            })
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

def get_ai_response(user_message):
    """Get real-time AI response using Google Gemini API"""
    
    # Try Gemini API first
    gemini_response = get_gemini_response(user_message)
    if gemini_response:
        return gemini_response
    
    # Fallback to intelligent response system
    return get_intelligent_response(user_message)

def get_gemini_response(user_message):
    """Get response from Google Gemini API"""
    try:
        # Get Gemini API key
        api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key or api_key == 'your-gemini-api-key-here':
            return None
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Use the correct model name
        model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            generation_config={
                'temperature': 0.7,
                'max_output_tokens': 500,
            }
        )
        
        # Create agricultural expert prompt
        prompt = f"""You are AgriGPT, an expert agricultural AI assistant. Provide specific, practical farming advice with exact details (fertilizer ratios, pesticide names, dosages, timing).

Question: {user_message}

Answer (be specific and actionable):"""
        
        # Generate response
        response = model.generate_content(prompt)
        
        if response and response.text:
            return response.text.strip()
        return None
            
    except Exception as e:
        print(f"Gemini Error: {e}")
        return None

def get_intelligent_response(user_message):
    """Intelligent response system with comprehensive agricultural knowledge"""
    msg = user_message.lower().strip()
    
    # Sugarcane pesticides - SPECIFIC
    if 'sugarcane' in msg and ('pesticide' in msg or 'pest' in msg):
        return "Sugarcane pesticides: 1) Early shoot borer - Chlorpyrifos 20% EC @ 2.5ml/L or Fipronil 5% SC @ 2ml/L, 2) Top borer - Cartap hydrochloride 50% SP @ 2g/L, 3) Termites - Chlorpyrifos 20% EC @ 4L/ha soil application, 4) Aphids - Imidacloprid 17.8% SL @ 0.3ml/L, 5) Whitefly - Thiamethoxam 25% WG @ 0.2g/L."
    
    # November crops - SPECIFIC
    if 'november' in msg and ('crop' in msg or 'plant' in msg or 'good' in msg):
        return "November crops (Rabi season): 1) Cereals - Wheat, Barley, Oats, 2) Pulses - Gram (chickpea), Lentil, Pea, 3) Oilseeds - Mustard, Rapeseed, Linseed, Safflower, 4) Vegetables - Cabbage, Cauliflower, Carrot, Radish, Spinach, Peas, Tomato, Brinjal, 5) Spices - Coriander, Fenugreek, Cumin. Best time for wheat sowing is 1st-15th November."
    
    # Advanced crop-specific responses
    crop_responses = {
        'tomato': {
            'fertilizer': "For tomatoes: Use NPK 19:19:19 @ 5g/plant initially. Switch to 13:40:13 during flowering (week 4-8) and 13:0:45 during fruiting. Apply calcium nitrate 2g/plant weekly to prevent blossom end rot. Add magnesium sulfate 1g/plant if leaves yellow.",
            'disease': "Tomato diseases: 1) Late blight - Mancozeb 75% WP @ 2g/L every 10 days, 2) Early blight - Chlorothalonil 75% WP @ 2g/L, 3) Bacterial wilt - use resistant varieties (Arka Rakshak, Pusa Ruby), 4) Leaf curl virus - control whiteflies with sticky traps.",
            'pest': "Tomato pests: 1) Fruit borer - Bt spray @ 2g/L weekly, 2) Aphids - Imidacloprid 17.8% SL @ 0.5ml/L, 3) Whiteflies - yellow sticky traps + neem oil 5ml/L, 4) Thrips - blue sticky traps.",
            'general': "Tomato cultivation: Plant in well-drained soil (pH 6.0-6.8), space 60x45cm, provide stakes 4-5 feet high. Water regularly but avoid waterlogging. Harvest when fruits turn pink-red."
        },
        'rice': {
            'fertilizer': "Rice fertilizer program: Apply 120:60:40 kg NPK/ha. Schedule: 50% N + full P&K as basal, 25% N at tillering (21 days), 25% N at panicle initiation (45 days). Add zinc sulfate 25kg/ha and gypsum 500kg/ha.",
            'disease': "Rice diseases: 1) Blast - Tricyclazole 75% WP @ 0.6g/L at boot leaf stage, 2) Sheath blight - Hexaconazole 5% SC @ 2ml/L, 3) Brown spot - Mancozeb 75% WP @ 2g/L, 4) Bacterial leaf blight - Copper oxychloride 50% WP @ 3g/L.",
            'pest': "Rice pests: 1) Stem borer - Cartap hydrochloride 4G @ 18.5kg/ha, 2) Brown planthopper - Imidacloprid 17.8% SL @ 0.4ml/L, 3) Leaf folder - Chlorantraniliprole 18.5% SC @ 0.4ml/L.",
            'general': "Rice cultivation: Transplant 25-30 day seedlings at 20x15cm spacing. Maintain 2-5cm standing water. Drain field 10 days before harvest when 80% grains turn golden yellow."
        },
        'wheat': {
            'fertilizer': "Wheat fertilizer: Apply 120:60:40 kg NPK/ha. Give 50% N + full P&K at sowing, 25% N at crown root initiation (21 days), 25% N at jointing stage (45 days).",
            'disease': "Wheat diseases: 1) Rust - Propiconazole 25% EC @ 1ml/L, 2) Bunt - Treat seeds with Tebuconazole 2% DS @ 1.5g/kg seed, 3) Powdery mildew - Sulfur 80% WP @ 3g/L.",
            'general': "Wheat sowing: Use 100kg seed/ha, sow in rows 20cm apart, 3-5cm deep. Irrigate at crown root initiation, tillering, jointing, flowering, and grain filling stages."
        }
    }
    
    # Check for specific crops
    for crop, responses in crop_responses.items():
        if crop in msg:
            if 'fertilizer' in msg or 'nutrition' in msg:
                return responses.get('fertilizer', responses['general'])
            elif 'disease' in msg or 'fungus' in msg:
                return responses.get('disease', responses['general'])
            elif 'pest' in msg or 'insect' in msg:
                return responses.get('pest', responses['general'])
            else:
                return responses['general']
    
    # Vegetable-specific responses
    if any(veg in msg for veg in ['potato', 'onion', 'cabbage', 'cauliflower', 'carrot']):
        if 'fertilizer' in msg:
            return "Vegetable fertilizer: Use NPK 19:19:19 @ 20-25g/m² as basal dose. Top dress with urea 10g/m² at 30 and 60 days. Apply compost 2-3kg/m² before planting."
        elif 'disease' in msg:
            return "Vegetable diseases: Use Mancozeb 75% WP @ 2g/L for fungal diseases, Copper oxychloride 50% WP @ 3g/L for bacterial diseases. Ensure good drainage and air circulation."
        else:
            return "Vegetable cultivation: Prepare raised beds, ensure good drainage, maintain soil pH 6.0-7.0, water regularly but avoid waterlogging, harvest at proper maturity."
    
    # Seasonal/timing questions
    if any(word in msg for word in ['when', 'time', 'season', 'month']):
        months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
        for month in months:
            if month in msg:
                return get_seasonal_advice(month)
        
        if 'kharif' in msg:
            return "Kharif season (June-October): Plant rice, cotton, sugarcane, maize, sorghum, pearl millet, finger millet, pulses (arhar, moong, urad), oilseeds (groundnut, sesame, castor)."
        elif 'rabi' in msg:
            return "Rabi season (November-April): Plant wheat, barley, oats, gram, pea, lentil, mustard, rapeseed, linseed, coriander, cumin, fenugreek."
    
    # Pest control
    if any(word in msg for word in ['pest', 'insect', 'aphid', 'caterpillar', 'borer']):
        return "Integrated Pest Management: 1) Monitor with pheromone traps, 2) Use beneficial insects (ladybird, lacewing), 3) Apply neem oil 5ml/L, 4) Chemical control: Imidacloprid for sucking pests, Bt spray for caterpillars, Chlorpyrifos for soil pests."
    
    # Disease management
    if any(word in msg for word in ['disease', 'fungus', 'blight', 'rot', 'wilt']):
        return "Disease management: 1) Use resistant varieties, 2) Seed treatment with Thiram 3g/kg, 3) Preventive spray: Mancozeb 75% WP @ 2g/L, 4) Curative spray: Propiconazole 25% EC @ 1ml/L, 5) Ensure proper drainage and spacing."
    
    # Soil management
    if 'soil' in msg:
        if 'ph' in msg:
            return "Soil pH management: For acidic soil (pH<6.0) - apply lime 2-4 tons/ha. For alkaline soil (pH>8.0) - apply gypsum 2-3 tons/ha + organic matter 15-20 tons/ha. Ideal pH for most crops: 6.0-7.0."
        else:
            return "Soil health improvement: 1) Add organic matter 15-20 tons/ha annually, 2) Practice crop rotation, 3) Use cover crops, 4) Apply biofertilizers (Rhizobium, PSB, KSB), 5) Maintain proper drainage, 6) Test soil every 2-3 years."
    
    # Fertilizer questions
    if 'fertilizer' in msg:
        return "Fertilizer application guide: 1) Soil test first, 2) Apply organic manure 15-20 tons/ha, 3) Use balanced NPK based on crop needs, 4) Split application - 50% basal, 25% at 30 days, 25% at 60 days, 5) Add micronutrients (Zn, B, Fe) as needed."
    
    # Water/irrigation
    if any(word in msg for word in ['water', 'irrigation', 'drought']):
        return "Water management: 1) Irrigate at critical growth stages, 2) Use drip irrigation (saves 40-60% water), 3) Apply mulch to reduce evaporation, 4) Check soil moisture at 15cm depth, 5) Water early morning or evening to reduce losses."
    
    # Organic farming
    if 'organic' in msg:
        return "Organic farming practices: 1) Use FYM 15-20 tons/ha, 2) Apply vermicompost 5 tons/ha, 3) Green manuring with dhaincha/sunhemp, 4) Biofertilizers (Rhizobium, Azotobacter, PSB), 5) Neem-based pesticides, 6) Crop rotation and intercropping."
    
    # Default intelligent response
    return "I'm your agricultural expert! I can provide specific advice on crop cultivation, fertilizer schedules, pest & disease management, soil health, irrigation, and seasonal farming practices. Please ask about specific crops or farming challenges you're facing."

def get_seasonal_advice(month):
    """Get seasonal farming advice for specific months"""
    seasonal_guide = {
        'january': "January farming: Harvest rabi crops, prepare for summer crops, plant vegetables (tomato, chili, brinjal), apply irrigation to standing crops, spray for aphids and thrips.",
        'february': "February activities: Continue rabi crop care, start summer crop preparation, plant fodder crops, begin nursery preparation for kharif vegetables, apply fertilizer to wheat at jointing stage.",
        'march': "March farming: Harvest early rabi crops, plant summer crops (fodder, vegetables), prepare fields for kharif season, start irrigation scheduling, control powdery mildew in crops.",
        'april': "April activities: Complete rabi harvest, plant summer vegetables, prepare nurseries for kharif crops, increase irrigation frequency, watch for pest buildup in hot weather.",
        'may': "May farming: Intensive summer crop care, prepare kharif crop seeds, deep summer plowing, maintain irrigation systems, control heat stress in crops with frequent light irrigation.",
        'june': "June activities: Begin kharif sowing (rice, cotton, sugarcane), transplant vegetable seedlings, ensure proper drainage for monsoon, apply pre-emergence herbicides.",
        'july': "July farming: Continue kharif planting, manage monsoon water, apply first top dressing to kharif crops, control weeds, watch for fungal diseases due to humidity.",
        'august': "August activities: Complete kharif sowing, apply second fertilizer dose, manage excess water, control pests and diseases, intercultural operations in standing crops.",
        'september': "September farming: Final kharif crop care, prepare for rabi season, control late season pests, begin rabi crop seed procurement, plan crop rotation.",
        'october': "October activities: Harvest early kharif crops, begin rabi land preparation, plant early rabi crops, control post-monsoon diseases, prepare irrigation systems.",
        'november': "November farming: Major rabi sowing month (wheat, gram, mustard), complete kharif harvest, apply basal fertilizers, ensure proper seed bed preparation.",
        'december': "December activities: Complete rabi sowing, apply irrigation to newly sown crops, control rabi season pests, plan for next year's cropping pattern, maintain farm equipment."
    }
    return seasonal_guide.get(month, "Please specify the month for seasonal farming advice.")

def get_fallback_response(user_message):
    """Main fallback function that uses intelligent responses"""
    return get_intelligent_response(user_message)