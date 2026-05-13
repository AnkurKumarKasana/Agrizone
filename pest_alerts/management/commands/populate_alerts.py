from django.core.management.base import BaseCommand
from pest_alerts.models import PestAlert

class Command(BaseCommand):
    help = 'Populate pest and disease alert data'

    def handle(self, *args, **options):
        alerts_data = [
            {
                'crop': 'rice',
                'pest_name': 'Brown Planthopper',
                'disease_name': '',
                'min_temp': 25,
                'max_temp': 35,
                'min_humidity': 70,
                'max_humidity': 100,
                'symptoms': 'Yellowing and drying of leaves from base upwards, stunted plant growth, hopper burn appearance with brown patches, reduced tillering, plants may die in severe cases',
                'prevention': 'Use resistant varieties (IR36, IR64), maintain proper plant spacing (20x15cm), avoid excessive nitrogen fertilizer, use light traps to monitor population, maintain 2-3cm water level, remove weeds regularly',
                'treatment': 'Spray Imidacloprid 17.8% SL @ 0.4ml/L or Buprofezin 25% SC @ 1.5ml/L. Alternate with Thiamethoxam 25% WG @ 0.2g/L. Apply 2-3 sprays at 10-day intervals. Ensure complete coverage of lower leaf surface',
                'severity': 'high'
            },
            {
                'crop': 'rice',
                'disease_name': 'Blast Disease',
                'pest_name': '',
                'min_temp': 20,
                'max_temp': 30,
                'min_humidity': 85,
                'max_humidity': 100,
                'symptoms': 'Diamond-shaped lesions with gray centers and brown margins on leaves, neck rot causing panicle breakage, node infection, seedling blight in nursery, yield loss up to 50%',
                'prevention': 'Use resistant varieties (Pusa Basmati 1121, IR64), avoid excessive nitrogen (>120kg/ha), ensure proper drainage, use disease-free seeds, maintain 2-3cm water depth, remove infected plant debris',
                'treatment': 'Spray Tricyclazole 75% WP @ 0.6g/L at tillering and boot leaf stage. Alternate with Carbendazim 50% WP @ 1g/L. Apply 2-3 sprays at 10-day intervals. For neck blast, spray at 5% flowering',
                'severity': 'critical'
            },
            {
                'crop': 'tomato',
                'disease_name': 'Late Blight',
                'pest_name': '',
                'min_temp': 15,
                'max_temp': 25,
                'min_humidity': 80,
                'max_humidity': 100,
                'symptoms': 'Water-soaked dark green lesions on leaves, white fungal growth on undersides, brown-black lesions on stems, fruit rot with greasy appearance, rapid spread in cool humid weather, complete crop loss possible',
                'prevention': 'Use disease-free certified seeds, maintain 60x45cm spacing, avoid overhead irrigation, use drip system, remove lower leaves, ensure good air circulation, destroy infected plants immediately',
                'treatment': 'Apply Mancozeb 75% WP @ 2g/L every 7-10 days preventively. For active infection use Metalaxyl 8% + Mancozeb 64% WP @ 2.5g/L. Alternate with Cymoxanil 8% + Mancozeb 64% WP. Apply 4-5 sprays',
                'severity': 'critical'
            },
            {
                'crop': 'wheat',
                'disease_name': 'Yellow Rust',
                'pest_name': '',
                'min_temp': 10,
                'max_temp': 20,
                'min_humidity': 60,
                'max_humidity': 100,
                'symptoms': 'Yellow-orange pustules in linear rows on leaves, premature leaf drying, reduced grain size and weight, shriveled grains, yield loss 30-70%, spreads rapidly in cool weather',
                'prevention': 'Use resistant varieties (HD 2967, PBW 343), timely sowing (Nov 1-15), balanced fertilization (120:60:40 NPK), avoid late sowing, remove volunteer wheat plants, crop rotation',
                'treatment': 'Spray Propiconazole 25% EC @ 1ml/L at first rust appearance. Alternate with Tebuconazole 25% EC @ 1ml/L. Apply 2 sprays at 15-day interval. Early application is critical',
                'severity': 'high'
            },
            {
                'crop': 'cotton',
                'pest_name': 'American Bollworm',
                'disease_name': '',
                'min_temp': 25,
                'max_temp': 35,
                'min_humidity': 50,
                'max_humidity': 80,
                'symptoms': 'Circular holes in leaves, damaged flower buds and bolls, presence of green-brown larvae, frass near entry holes, boll shedding, reduced lint quality, yield loss 30-60%',
                'prevention': 'Use pheromone traps (12-15/ha), intercrop with marigold/coriander, install bird perches, hand-pick larvae, remove damaged bolls, deep summer plowing, use Bt cotton varieties',
                'treatment': 'Spray Chlorantraniliprole 18.5% SC @ 0.4ml/L or Emamectin benzoate 5% SG @ 0.4g/L. Alternate with Spinosad 45% SC @ 0.3ml/L. Apply 3-4 sprays at 10-day intervals during flowering',
                'severity': 'high'
            },
            {
                'crop': 'sugarcane',
                'pest_name': 'Early Shoot Borer',
                'disease_name': '',
                'min_temp': 20,
                'max_temp': 35,
                'min_humidity': 60,
                'max_humidity': 90,
                'symptoms': 'Dead hearts in young plants (1-3 months), wilting of central shoots, presence of bore holes with frass, stunted growth, reduced tillering, yield loss 20-30%',
                'prevention': 'Use healthy disease-free setts, treat setts with Chlorpyrifos before planting, avoid ratoon crops, intercrop with onion/garlic, remove and destroy dead hearts, maintain field sanitation',
                'treatment': 'Apply Chlorpyrifos 20% EC @ 2.5ml/L or Fipronil 5% SC @ 2ml/L in leaf whorls. Use Cartap hydrochloride 50% SP @ 2g/L. Apply 2-3 treatments at 15-day intervals in early crop stage',
                'severity': 'medium'
            },
            {
                'crop': 'maize',
                'pest_name': 'Fall Armyworm',
                'disease_name': '',
                'min_temp': 20,
                'max_temp': 35,
                'min_humidity': 50,
                'max_humidity': 90,
                'symptoms': 'Irregular holes in leaves, window-pane feeding, frass in leaf whorls, damaged tassels and ears, presence of green-brown larvae with inverted Y on head, severe defoliation',
                'prevention': 'Early sowing, use pheromone traps, intercrop with pulses, hand-pick egg masses and larvae, use bird perches, apply neem-based products, maintain field hygiene',
                'treatment': 'Spray Chlorantraniliprole 18.5% SC @ 0.4ml/L or Spinetoram 11.7% SC @ 0.5ml/L in leaf whorls. Use Emamectin benzoate 5% SG @ 0.4g/L. Apply 2-3 sprays at 7-10 day intervals',
                'severity': 'critical'
            },
            {
                'crop': 'potato',
                'disease_name': 'Late Blight',
                'pest_name': '',
                'min_temp': 10,
                'max_temp': 25,
                'min_humidity': 80,
                'max_humidity': 100,
                'symptoms': 'Water-soaked lesions on leaves, white fungal growth on undersides, brown-black stem lesions, tuber rot with reddish-brown discoloration, foul smell, rapid disease spread, total crop loss possible',
                'prevention': 'Use certified disease-free seed tubers, earthing up properly, avoid overhead irrigation, maintain 60x20cm spacing, remove volunteer plants, destroy infected plants, use resistant varieties',
                'treatment': 'Spray Metalaxyl 8% + Mancozeb 64% WP @ 2.5g/L preventively every 7-10 days. For active infection use Cymoxanil 8% + Mancozeb 64% WP @ 2g/L. Apply 5-6 sprays during crop season',
                'severity': 'critical'
            },
        ]
        
        PestAlert.objects.all().delete()
        
        for data in alerts_data:
            PestAlert.objects.create(**data)
            self.stdout.write(f"Created alert: {data['crop']} - {data.get('pest_name') or data.get('disease_name')}")
        
        self.stdout.write(self.style.SUCCESS(f'Successfully populated {len(alerts_data)} pest alerts'))
