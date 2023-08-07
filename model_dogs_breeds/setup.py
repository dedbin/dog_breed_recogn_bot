import pandas as pd
import os
from matplotlib import image as mpimg
import matplotlib.pyplot as plt

labels = pd.read_csv(r'C:\kaggle\input\dog-breed-ingentification\labels.csv')  # загрузка названий пород песиков

train_dir = r'C:\kaggle\input\dog-breed-ingentification\train'  # путь к папке с тренировочными изображениями
test_dir = r'C:\kaggle\input\dog-breed-ingentification\test'  # путь к папке с тестовыми изображениями

img_size = 224  # размер изображений
batch_size = 16  # размер батча

breeds = sorted(labels['breed'].unique())
num_classes = len(breeds)
breed_to_label = {breed: i for i, breed in enumerate(breeds)}
translations = {
    'affenpinscher': 'Аффенпинчер',
    'afghan_hound': 'Афганская борзая',
    'african_hunting_dog': 'Африканская охотничья собака',
    'airedale': 'Ирландский лаурель',
    'american_staffordshire_terrier': 'Американский стаффордширский терьер',
    'appenzeller': 'Аппенцеллер',
    'australian_terrier': 'Австралийский терьер',
    'basenji': 'Басенджи',
    'basset': 'Бассет',
    'beagle': 'Бигль',
    'bedlington_terrier': 'Бедлингтон-терьер',
    'bernese_mountain_dog': 'Бернский зенненхунд',
    'black-and-tan_coonhound': 'Черно-пегая кунхаунд',
    'blenheim_spaniel': 'Бленхеймский спаниель',
    'bloodhound': 'Гончая',
    'bluetick': 'Голубая ложная гончая',
    'border_collie': 'Бордер-колли',
    'border_terrier': 'Бордер-терьер',
    'borzoi': 'Борзая',
    'boston_bull': 'Бостонский терьер',
    'bouvier_des_flandres': 'Фландрский бувье',
    'boxer': 'Боксер',
    'brabancon_griffon': 'Брабансонский гриффон',
    'briard': 'Бриар',
    'brittany_spaniel': 'Бриттанский спаниель',
    'bull_mastiff': 'Бульмастиф',
    'cairn': 'Кейрн-терьер',
    'cardigan': 'Кардиган-уэльский корги',
    'chesapeake_bay_retriever': 'Чесапик-бэй-ретривер',
    'chihuahua': 'Чихуахуа',
    'chow': 'Чау-чау',
    'clumber': 'Кламбер-спаниель',
    'cocker_spaniel': 'Кокер-спаниель',
    'collie': 'Колли',
    'curly-coated_retriever': 'Лабрадор-ретривер с кудрявой шерстью',
    'dandie_dinmont': 'Дэнди-динмонт',
    'dhole': 'Дхоль',
    'dingo': 'Динго',
    'doberman': 'Доберман',
    'english_foxhound': 'Английская лисичья гончая',
    'english_setter': 'Английский сеттер',
    'english_springer': 'Английский спрингер-спаниель',
    'entlebucher': 'Энтльбухер',
    'eskimo_dog': 'Эскимосская собака',
    'flat-coated_retriever': 'Лабрадор-ретривер с гладкой шерстью',
    'french_bulldog': 'Французский бульдог',
    'german_shepherd': 'Немецкая овчарка',
    'german_short-haired_pointer': 'Немецкий курцхаар',
    'giant_schnauzer': 'Джайант шнауцер',
    'golden_retriever': 'Золотистый ретривер',
    'gordon_setter': 'Гордон сеттер',
    'great_dane': 'Великий датский дог',
    'great_pyrenees': 'Пиренейская горная собака',
    'greater_swiss_mountain_dog': 'Большая швейцарская горная собака',
    'groenendael': 'Гронендаль',
    'ibizan_hound': 'Ибизский борзая',
    'irish_setter': 'Ирландский сеттер',
    'irish_terrier': 'Ирландский терьер',
    'irish_water_spaniel': 'Ирландский водяной спаниель',
    'irish_wolfhound': 'Ирландский волкодав',
    'italian_greyhound': 'Итальянский грейхаунд',
    'japanese_spaniel': 'Японский спаниель',
    'keeshond': 'Кеесхонд',
    'kelpie': 'Австралийская овчарка (келпи)',
    'kerry_blue_terrier': 'Керри-блю терьер',
    'komondor': 'Комондор',
    'kuvasz': 'Кувас',
    'labrador_retriever': 'Лабрадор ретривер',
    'lakeland_terrier': 'Лейкленд терьер',
    'leonberg': 'Леонберг',
    'lhasa': 'Лхаса апсо',
    'malamute': 'Маламут',
    'malinois': 'Малинуа',
    'maltese_dog': 'Мальтийская болонка',
    'mexican_hairless': 'Мексиканская голая собака',
    'miniature_pinscher': 'Миниатюрный пинчер',
    'miniature_poodle': 'Миниатюрный пудель',
    'miniature_schnauzer': 'Миниатюрный шнауцер',
    'newfoundland': 'Ньюфаундленд',
    'norfolk_terrier': 'Норфолк терьер',
    'norwegian_elkhound': 'Норвежский эльхунд',
    'norwich_terrier': 'Норвич терьер',
    'old_english_sheepdog': 'Бобтейл',
    'otterhound': 'Оттерхаунд',
    'papillon': 'Папильон',
    'pekinese': 'Пекинес',
    'pembroke': 'Пемброк-уэльский корги',
    'pomeranian': 'Померанский шпиц',
    'pug': 'Мопс',
    'redbone': 'Редбоун',
    'rhodesian_ridgeback': 'Родезийский риджбек',
    'rottweiler': 'Ротвейлер',
    'saint_bernard': 'Сенбернар',
    'saluki': 'Салюки',
    'samoyed': 'Самоедская собака',
    'schipperke': 'Шипперке',
    'scotch_terrier': 'Скотч-терьер',
    'scottish_deerhound': 'Шотландский дирхаунд',
    'sealyham_terrier': 'Силихэм-терьер',
    'shetland_sheepdog': 'Шелти',
    'shih-tzu': 'Ших-тцу',
    'siberian_husky': 'Сибирский хаски',
    'silky_terrier': 'Силки терьер',
    'soft-coated_wheaten_terrier': 'Пшеничный терьер',
    'staffordshire_bullterrier': 'Стаффордширский бультерьер',
    'standard_poodle': 'Стандартный пудель',
    'standard_schnauzer': 'Средний шнауцер',
    'sussex_spaniel': 'Сассекс-спаниель',
    'tibetan_mastiff': 'Тибетский мастиф',
    'tibetan_terrier': 'Тибетский терьер',
    'toy_poodle': 'Той пудель',
    'toy_terrier': 'Той-терьер',
    'vizsla': 'Венгерская выжла',
    'walker_hound': 'Уолкер хаунд',
    'weimaraner': 'Веймаранер',
    'welsh_springer_spaniel': 'Уэльский спрингер-спаниель',
    'west_highland_white_terrier': 'Вест-хайленд-уайт-терьер',
    'whippet': 'Виппет',
    'wire-haired_fox_terrier': 'Жесткошерстный фокстерьер',
    'yorkshire_terrier': 'Йоркширский терьер'
}


if __name__ == '__main__':
    # показать изображения для обучения
    '''for i in range(5):
        filename = labels.iloc[i]['id'] + '.jpg'
        label = labels.iloc[i]['breed']

        img_path = os.path.join(train_dir, filename)
        img = mpimg.imread(img_path)
        plt.imshow(img)
        plt.title(label)
        plt.show()'''

    # показать изображения для теста
    '''for i in range(5):
        filename = os.listdir(test_dir)[i]
        img_path = os.path.join(test_dir, filename)
        img = mpimg.imread(img_path)
        plt.imshow(img)
        plt.title(filename)
        plt.show()'''
    print(f'{breeds = }')
    print(f'{translations = }')
    translations.keys()
    tr = ['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale', 'american_staffordshire_terrier', 'appenzeller', 'australian_terrier', 'basenji', 'basset', 'beagle', 'bedlington_terrier', 'bernese_mountain_dog', 'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound', 'bluetick', 'border_collie', 'border_terrier', 'borzoi', 'boston_bull', 'bouvier_des_flandres', 'boxer', 'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff', 'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua', 'chow', 'clumber', 'cocker_spaniel', 'collie', 'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo', 'doberman', 'english_foxhound', 'english_setter', 'english_springer', 'entlebucher', 'eskimo_dog', 'flat-coated_retriever', 'french_bulldog', 'german_shepherd', 'german_short-haired_pointer', 'giant_schnauzer', 'golden_retriever', 'gordon_setter', 'great_dane', 'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael', 'ibizan_hound', 'irish_setter', 'irish_terrier', 'irish_water_spaniel', 'irish_wolfhound', 'italian_greyhound', 'japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier', 'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier', 'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog', 'mexican_hairless', 'miniature_pinscher', 'miniature_poodle', 'miniature_schnauzer', 'newfoundland', 'norfolk_terrier', 'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog', 'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian', 'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler', 'saint_bernard', 'saluki', 'samoyed', 'schipperke', 'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier', 'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier', 'soft-coated_wheaten_terrier', 'staffordshire_bullterrier', 'standard_poodle', 'standard_schnauzer', 'sussex_spaniel', 'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier', 'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel', 'west_highland_white_terrier', 'whippet', 'wire-haired_fox_terrier', 'yorkshire_terrier']
    br = ['Аффенпинчер', 'Афганская борзая', 'Африканская охотничья собака', 'Ирландский лаурель', 'Американский стаффордширский терьер', 'Аппенцеллер зенненхунд', 'Австралийский терьер', 'Басенджи', 'Бассет', 'Бигль', 'Бедлингтон-терьер', 'Бернский зенненхунд', 'Черно-пегая кунхаунд', 'Бленхеймский спаниель', 'Гончая', 'Синепалый енотовидный пес', 'Бордер-колли', 'Бордер-терьер', 'Борзая', 'Бостонский бык', 'Фландрский бувье', 'Немецкий боксёр', 'Брабансонский гриффон', 'Бриар', 'Бриттанский спаниель', 'Бульмастиф', 'Кейрн-терьер', 'Кардиган-уэльский корги', 'Чесапик-бэй-ретривер', 'Чихуахуа', 'Чау-чау', 'Кламбер-спаниель', 'Кокер-спаниель', 'Колли', 'Лабрадор-ретривер с кудрявой шерстью', 'Дэнди-динмонт', 'Дхоль', 'Динго', 'Доберман', 'Английская лисичья гончая', 'Английский сеттер', 'Английский спрингер-спаниель', 'Энтльбухер', 'Эскимосская собака', 'Лабрадор-ретривер с гладкой шерстью', 'Французский бульдог', 'Немецкая овчарка', 'Курцхаар', 'Ризеншнауцер', 'Золотистый ретривер', 'Шотландский сеттер', 'Великий датский дог', 'Пиренейская горная собака', 'Большая швейцарская горная собака', 'Бельгийская овчарка Грюнендаль', 'Ибизский борзая', 'Ирландский сеттер', 'Ирландский терьер', 'Ирландский водяной спаниель', 'Ирландский волкодав', 'Итальянский грейхаунд', 'Японский спаниель', 'Кеесхонд', 'Австралийский келпи']