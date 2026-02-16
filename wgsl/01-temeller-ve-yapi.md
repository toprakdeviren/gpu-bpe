---
title: Temeller ve Yapı
section: "1-5, 16"
source: "W3C WGSL Spec §1–§5, §16"
---

# 1. Temeller ve Yapı

> Dilin iskeleti: giriş, modül yapısı, sözdizimi kuralları, direktifler, kapsam ve keyword/token özeti.

---

## §1 Introduction

WebGPU Shading Language (WGSL), [WebGPU](https://www.w3.org/TR/webgpu/) için tasarlanmış shader dilidir. WebGPU API kullanan uygulamalar, GPU üzerinde çalışan shader programlarını WGSL ile ifade eder.

```wgsl
// Nokta ışıklarıyla textured geometriyi aydınlatan bir fragment shader.

// Storage buffer binding'den gelen ışıklar.
struct PointLight {
  position : vec3f,
  color : vec3f,
}

struct LightStorage {
  pointCount : u32,
  point : array<PointLight>,
}
@group(0) @binding(0) var<storage> lights : LightStorage;

// Texture ve sampler.
@group(1) @binding(0) var baseColorSampler : sampler;
@group(1) @binding(1) var baseColorTexture : texture_2d<f32>;

// Fonksiyon argümanları vertex shader'dan gelen değerlerdir.
@fragment
fn fragmentMain(@location(0) worldPos : vec3f,
                @location(1) normal : vec3f,
                @location(2) uv : vec2f) -> @location(0) vec4f {
  // Yüzeyin temel rengini texture'dan örnekle.
  let baseColor = textureSample(baseColorTexture, baseColorSampler, uv);

  let N = normalize(normal);
  var surfaceColor = vec3f(0);

  // Sahnedeki nokta ışıklarını döngüle.
  for (var i = 0u; i < lights.pointCount; i++) {
    let worldToLight = lights.point[i].position - worldPos;
    let dist = length(worldToLight);
    let dir = normalize(worldToLight);

    // Bu ışığın yüzey rengine katkısını belirle.
    let radiance = lights.point[i].color * (1 / pow(dist, 2));
    let nDotL = max(dot(N, dir), 0);

    // Yüzey rengine ışık katkısını biriktir.
    surfaceColor += baseColor.rgb * radiance * nDotL;
  }

  // Birikmiş yüzey rengini döndür.
  return vec4(surfaceColor, baseColor.a);
}
```

### 1.1 Overview

WebGPU, GPU'ya bir iş birimi **GPU command** formunda gönderir. WGSL iki tür GPU komutuyla ilgilenir:

- **draw command** — Bir [render pipeline](#gpurenderpipeline)'ı input, output ve bağlı resource'lar bağlamında çalıştırır.
- **dispatch command** — Bir [compute pipeline](#gpucomputepipeline)'ı input ve bağlı resource'lar bağlamında çalıştırır.

Her iki pipeline türü de WGSL ile yazılmış shader'ları kullanır.

**Shader**, bir pipeline'daki shader stage'i çalıştıran WGSL programının parçasıdır. Bir shader şunlardan oluşur:

- Bir **entry point** fonksiyonu.
- Entry point'ten başlayarak çağrılan tüm fonksiyonların transitif kapanışı (hem **user-defined** hem **built-in** fonksiyonlar dahil).
- Bu fonksiyonlar tarafından **statically accessed** edilen değişkenler ve sabitler kümesi.
- Tüm bu fonksiyonları, değişkenleri ve sabitleri tanımlamak veya analiz etmek için kullanılan tipler kümesi.

> **Not:** Bir WGSL programı entry point gerektirmez; ancak API tarafından çalıştırılamaz çünkü `GPUProgrammableStage` oluşturmak için entry point zorunludur.

Bir shader stage çalıştırılırken implementasyon:

1. Module-scope'ta bildirilen sabitlerin değerlerini hesaplar.
2. Shader'ın **resource interface**'indeki değişkenlere resource'ları bağlar.
3. Diğer module-scope değişkenler için bellek ayırır ve belirtilen başlangıç değerleriyle doldurur.
4. Entry point'in formal parametrelerini (varsa) shader stage input'larıyla doldurur.
5. Entry point'in return value'sunu (varsa) shader stage output'larına bağlar.
6. Ardından entry point'i çağırır (invoke).

#### WGSL Programının Organizasyonu

Bir WGSL programı şu öğelerden oluşur:

- **Directives** — Modül düzeyinde davranış kontrolleri.
- **Functions** — Çalışma davranışını belirtir.
- **Statements** — Bildirimler veya çalıştırılabilir davranış birimleri.
- **Literals** — Saf matematiksel değerler için metin gösterimleri.
- **Constants** — Belirli bir zamanda hesaplanan bir değer için isim sağlar.
- **Variables** — Bir değer tutan bellek için isim sağlar.
- **Expressions** — Bir değerler kümesini birleştirerek bir sonuç değeri üretir.
- **Types** — Her biri şunları tanımlar:
  - Bir değerler kümesi.
  - Desteklenen ifadeler üzerindeki kısıtlamalar.
  - Bu ifadelerin semantiği.
- **Attributes** — Ek bilgi belirtmek için objeleri modifiye eder (interface'ler, diagnostic filter'lar vb.).

> **Not:** Bir WGSL programı şu anda tek bir WGSL modülünden oluşur.

#### İmperatif Dil Yapısı

WGSL imperatif bir dildir: davranış, çalıştırılacak statement'ların bir dizisi olarak belirtilir. Statement'lar:

- Sabit veya değişken bildirebilir.
- Değişkenlerin içeriğini modifiye edebilir.
- Yapılandırılmış programlama yapılarıyla çalışma sırasını değiştirebilir:
  - **Seçici çalıştırma:** `if` (opsiyonel `else if` ve `else` ile), `switch`.
  - **Tekrarlama:** `loop`, `while`, `for`.
  - **İç içe yapıdan çıkış:** `continue`, `break`, `break if`.
  - **Refactoring:** fonksiyon çağrısı ve `return`.
- İfadeleri değerlendirerek değer hesaplayabilir.
- Shader creation time'da constant expression'lar üzerinde varsayımları kontrol edebilir.

#### Tip Sistemi Genel Bakış

WGSL **statik tipli** bir dildir: her ifadenin ürettiği değerin tipi, yalnızca program kaynağı incelenerek belirlenir.

WGSL şu tipleri içerir:
- **boolean** ve **sayısal** tipler (integer, floating point).
- **Composite** tipler: vector, matrix, array, structure.
- **Özel** tipler: atomic (benzersiz operasyonlar sağlar).
- **Memory view** tipleri: bellekte saklanabilen tipleri tanımlar.
- **Texture ve sampler** tipleri: yaygın GPU rendering donanımını açığa çıkarır.

WGSL concrete tiplerden implicit dönüşüm veya promotion yapmaz, ancak **abstract** tiplerden implicit dönüşüm ve promotion sağlar. Concrete bir sayısal veya boolean tipten diğerine dönüşüm, açık bir value constructor, conversion veya `bitcast` gerektirir.

#### Invocation'lar ve Paralel Çalışma

Bir shader stage'in işi bir veya daha fazla **invocation**'a bölünür. Her invocation, entry point'i biraz farklı koşullar altında çalıştırır.

Invocation'ların değişken paylaşımı:
- Tüm invocation'lar shader interface'indeki resource'ları paylaşır.
- Compute shader'da aynı **workgroup** içindeki invocation'lar `workgroup` address space'deki değişkenleri paylaşır. Farklı workgroup'lardaki invocation'lar bu değişkenleri paylaşmaz.

Her invocation kendi bağımsız bellek alanına sahiptir (`private` ve `function` address space'leri).

Bir shader stage içindeki invocation'lar eşzamanlı çalışır ve genellikle paralel çalışabilir. Shader yazarı şunları sağlamaktan sorumludur:
- Texture sampling ve control barrier gibi ilkel operasyonların **uniformity** gereksinimlerini karşılama.
- Paylaşılan değişkenlere çakışan erişimleri koordine ederek data race'lerden kaçınma.

#### Davranışsal Gereksinimler

WGSL bazen belirli bir özellik için birden fazla olası davranışa izin verir — bu bir taşınabilirlik tehlikesidir. **Behavioral requirements**, implementasyonun bir WGSL programını işlerken veya çalıştırırken gerçekleştireceği eylemlerdir.

### 1.2 Syntax Notation

WGSL'in sözdizimsel grammar'ının kuralları şu gösterimlerle ifade edilir:

| Gösterim | Anlam |
|----------|-------|
| *İtalik metin* | Sözdizimi kuralı (syntax rule) |
| **`'bold mono'`** | Keyword veya token |
| **`:`** | Sözdizimi kuralı kaydı |
| **`\|`** | Alternatifler |
| **`?`** | Önceki öğe sıfır veya bir kez oluşur (opsiyonel) |
| **`*`** | Önceki öğe sıfır veya daha fazla kez oluşur |
| **`+`** | Önceki öğe bir veya daha fazla kez oluşur |
| **`( )`** | Öğelerin gruplanması |

### 1.3 Mathematical Terms and Notation

#### Açılar (Angles)

- Açılar radyan cinsinden ölçülür.
- Referans ışını: orijinden (0,0) (+∞,0) yönüne doğru olan ışın.
- θ açısı, karşılaştırma ışını saat yönünün tersine hareket ettikçe artar.
- Tam bir dairede 2π radyan vardır.

| Açı | Yön |
|-----|-----|
| 0 | Sağa → (1,0) |
| π/4 | (1,1) yönü |
| π/2 | Yukarı → (0,1) |
| π | Sola → (-1,0) |
| 3π/2 | Aşağı → (0,-1) |
| 2π | Sağa → (1,0) |

#### Hiperbolik Açı (Hyperbolic Angle)

Hiperbolik açı, geleneksel anlamda bir açı değil, birimsiz bir alandır:
- *x*² - *y*² = 1 hiperbolünü (*x* > 0 için) düşünün.
- Orijinden hiperbol üzerindeki bir noktaya *R* ışını çizin.
- *a*, *R* ışını, *x* ekseni ve hiperbol eğrisi tarafından çevrelenen alanın iki katıdır.
- *R* yukarıda ise *a* pozitif, aşağıda ise negatiftir.
- Bu durumda *x* = cosh(*a*) ve *y* = sinh(*a*).

#### Sonsuzluklar ve Genişletilmiş Reel Sayılar

- **Pozitif sonsuzluk** (+∞): Tüm reel sayılardan kesinlikle büyük benzersiz değer.
- **Negatif sonsuzluk** (−∞): Tüm reel sayılardan kesinlikle küçük benzersiz değer.
- **Extended real** sayılar: Reel sayılar ∪ {+∞, −∞}. Bilgisayarlar bunları yaklaşık olarak temsil etmek için floating point tipler kullanır (bkz. [§15.7 Floating Point Evaluation](06-paralel-calisma-ve-dogruluk.md#157-floating-point-evaluation)).

#### Aralıklar (Intervals)

Bir **interval**, alt ve üst sınırı olan sürekli bir sayı kümesidir:

| Gösterim | Tanım |
|----------|-------|
| [*a*, *b*] | *a* ≤ *x* ≤ *b* (kapalı aralık) |
| [*a*, *b*) | *a* ≤ *x* < *b* (yarı açık) |
| (*a*, *b*] | *a* < *x* ≤ *b* (yarı açık) |

#### Matematiksel Fonksiyonlar

**Floor (taban) ifadesi** — extended real sayılar için:
- ⌊+∞⌋ = +∞
- ⌊−∞⌋ = −∞
- Reel *x* için: ⌊*x*⌋ = *k*, burada *k* ≤ *x* < *k*+1 olan tek tamsayı

**Ceiling (tavan) ifadesi** — extended real sayılar için:
- ⌈+∞⌉ = +∞
- ⌈−∞⌉ = −∞
- Reel *x* için: ⌈*x*⌉ = *k*, burada *k*-1 < *x* ≤ *k* olan tek tamsayı

**Truncate (kesme) fonksiyonu** — extended real sayılar için:
- truncate(+∞) = +∞
- truncate(−∞) = −∞
- Reel *x* için: mutlak değeri *x*'in mutlak değerine eşit veya küçük olan en yakın tam sayı.
  - truncate(*x*) = ⌊*x*⌋ eğer *x* ≥ 0, ⌈*x*⌉ eğer *x* < 0.

**roundUp fonksiyonu** — pozitif tamsayılar *k* ve *n* için:
- roundUp(*k*, *n*) = ⌈*n* ÷ *k*⌉ × *k*

**Transpose (devrik)** — *c*-sütun *r*-satır matrisi *A* için:
- transpose(*A*) = *A*ᵀ
- transpose(*A*)ᵢ,ⱼ = *A*ⱼ,ᵢ
- Bir sütun vektörünün devriği, sütun vektörünü 1-satırlı matris olarak yorumlayarak tanımlanır (satır vektörü için tersi geçerlidir).

---

## §2 WGSL Module

Bir WGSL programı tek bir WGSL modülünden oluşur.

Modül, opsiyonel **directive**'lerin ardından gelen module scope **declaration**'lar ve **assertion**'lardan oluşan bir dizidir. Modül şu öğelere ayrılır:

- **Directives** — Modül düzeyinde davranış kontrolleri.
- **Functions** — Çalışma davranışını belirtir.
- **Statements** — Bildirimler veya çalıştırılabilir davranış birimleri.
- **Literals** — Saf matematiksel değerler için metin gösterimleri.
- **Variables** — Bir değer tutan bellek için isim sağlar.
- **Constants** — Belirli bir zamanda hesaplanan bir değer için isim sağlar.
- **Expressions** — Bir değerler kümesini birleştirerek sonuç değeri üretir.
- **Types** — Değer kümesi, desteklenen ifade kısıtlamları ve semantiği tanımlar.
- **Attributes** — Entry point interface'leri ve diagnostic filter'lar gibi ek bilgi belirtir.

```bnf
translation_unit :
  global_directive* (global_decl | global_assert | ';')*

global_decl :
  | global_variable_decl ';'
  | global_value_decl ';'
  | type_alias_decl ';'
  | struct_decl
  | function_decl
```

### 2.1 Shader Lifecycle

WGSL programının ve shader'larının yaşam döngüsünde dört anahtar olay vardır:

1. **Shader module creation** — WebGPU `createShaderModule()` metodu çağrıldığında gerçekleşir. WGSL programının kaynak metni bu aşamada sağlanır.

2. **Pipeline creation** — WebGPU `createComputePipeline()` veya `createRenderPipeline()` metodu çağrıldığında gerçekleşir. Entry point'in shader'ını oluşturan kod dikkate alınır; entry point ile ilgisi olmayan kod derleme öncesinde etkili bir şekilde atılır.
   > **Not:** Her shader stage ayrı olarak derlenir ve dolayısıyla modülün farklı bölümlerini içerebilir.

3. **Shader execution start** — Bir draw veya dispatch komutu GPU'ya iletildiğinde, pipeline çalıştırılmaya başlandığında ve shader stage entry point fonksiyonu çağrıldığında gerçekleşir.

4. **Shader execution end** — Shader'daki tüm iş tamamlandığında gerçekleşir:
   - Tüm invocation'lar sonlanır.
   - Resource'lara tüm erişimler tamamlanır.
   - Çıktılar (varsa) downstream pipeline stage'lerine iletilir.

Olayların sıralaması şunlara bağlıdır:
- **Veri bağımlılıkları:** Shader çalıştırma bir pipeline, pipeline ise bir shader modülü gerektirir.
- **Nedensellik:** Shader'ın bitmeden önce çalışmaya başlaması gerekir.

### 2.2 Errors

Bir WebGPU implementasyonu iki nedenden dolayı shader'ı işleyemeyebilir:

- **Program error** — Shader, WGSL veya WebGPU spesifikasyonlarının gereksinimlerini karşılamıyorsa oluşur.
- **Uncategorized error** — Tüm gereksinimler karşılansa bile oluşabilir. Olası nedenler:
  - Shader'lar çok karmaşık olup implementasyonun kapasitesini aşıyor (öngörülmüş limitlerle kolayca yakalanmayan şekilde).
  - WebGPU implementasyonunda bir kusur.

Bir işleme hatası shader yaşam döngüsünde üç aşamada oluşabilir:

| Hata Türü | Tespit Zamanı | Açıklama |
|-----------|---------------|----------|
| **shader-creation error** | `createShaderModule()` zamanı | Yalnızca WGSL modül kaynak metnine ve `createShaderModule` API'sine mevcut bilgilere bağlıdır. Spec'te *must* kullanılan ifadeler ihlal edilirse tetiklenir. |
| **pipeline-creation error** | `createComputePipeline()` / `createRenderPipeline()` zamanı | WGSL modül kaynak metni ve pipeline creation API'sine mevcut bilgilere bağlıdır. Yalnız derlenen entry point'in shader'ındaki kod için tetiklenir. |
| **dynamic error** | Shader çalışma zamanı | Tespit edilebilir veya edilemeyebilir (ör. data race tespit edilemeyebilir). |

Her gereksinim mümkün olan en erken fırsatta kontrol edilir:
- Shader-creation zamanında tespit edilebilen bir gereksinim → shader-creation error
- Pipeline-creation zamanında tespit edilebilen ama daha erken tespit edilemeyen → pipeline-creation error

**Hataların sonuçları:**
- Shader-creation veya pipeline-creation error'lu bir modül pipeline'a dahil edilmez ve çalıştırılmaz.
- Tespit edilebilir hatalar bir **diagnostic** tetikler.
- Dynamic error oluşursa: bellek erişimleri shader stage input/output'ları, bağlı resource'lar ve modüldeki diğer değişkenlerle sınırlandırılır. Aksi takdirde program spec'te açıklandığı gibi davranmayabilir (etkileri non-local olabilir).

### 2.3 Diagnostics

İmplementasyon, shader module creation veya pipeline creation sırasında **diagnostic** (teşhis mesajları) üretebilir. Diagnostic, uygulama yazarının yararına üretilen bir mesajdır.

Bir diagnostic, belirli bir koşul karşılandığında **triggered** (tetiklenir). Bu koşul **triggering rule** olarak bilinir. Koşulun karşılandığı kaynak metin konumu **triggering location** olarak adlandırılır.

Bir diagnostic'in özellikleri:
- Bir **severity** (önem derecesi)
- Bir **triggering rule**
- Bir **triggering location**

**Severity seviyeleri** (büyükten küçüğe):

| Severity | Açıklama |
|----------|----------|
| **error** | Hata. shader-creation error veya pipeline-creation error'a karşılık gelir. |
| **warning** | Uyarı. Hata değildir ama uygulama geliştiricisinin dikkatini gerektiren bir anomali. |
| **info** | Bilgi. Hata veya uyarı değildir ama dikkat gerektiren kayda değer bir durum. |
| **off** | Devre dışı. Uygulamaya iletilmez. |

Triggering rule isimleri ya tek bir `diagnostic_name_token` ya da nokta (`.`) ile ayrılmış iki `diagnostic_name_token` olarak belirtilir.

```bnf
diagnostic_rule_name :
  | diagnostic_name_token
  | diagnostic_name_token '.' diagnostic_name_token
```

#### 2.3.1 Diagnostic Processing

Tetiklenen diagnostic'ler şu şekilde işlenir:

1. Her diagnostic *D* için, *D*'nin triggering location'ını kapsayan en küçük affected range'e sahip ve aynı triggering rule'a sahip diagnostic filter bulunur.
   - Filter varsa → *D*'nin severity'si güncellenir.
   - Yoksa → *D* değişmez.
2. Severity `off` olan diagnostic'ler atılır.
3. En az bir `info` severity'li diagnostic varsa → aynı triggering rule'a sahip diğer `info` diagnostic'ler atılabilir.
4. En az bir `warning` severity'li diagnostic varsa → aynı triggering rule'a sahip `info` veya `warning` diagnostic'ler atılabilir.
5. En az bir `error` severity'li diagnostic varsa:
   - Diğer diagnostic'ler (dahil `error` olanlar) atılabilir.
   - Bir **program error** üretilir (shader-creation veya pipeline-creation error).
6. Shader module creation sırasında → diagnostic'ler WebGPU `GPUCompilationInfo.messages`'ı doldurur.
7. Pipeline creation sırasında → `error` diagnostic'ler `GPUProgrammableStage` doğrulamasında hatayla sonuçlanır.

> **Not:** Kurallar, implementasyonun bir hata tespit edilir edilmez WGSL modülü işlemeyi durdurmasına izin verir. Farklı implementasyonlar aynı WGSL modülü için farklı diagnostic örnekleri raporlayabilir.

#### 2.3.2 Filterable Triggering Rules

Çoğu diagnostic koşulsuz olarak raporlanır. Bazıları ise triggering rule'ları isimlendirilerek **filtrelenebilir**.

| Filterable Triggering Rule | Default Severity | Açıklama |
|---------------------------|------------------|----------|
| `derivative_uniformity` | `error` | Derivative hesaplayan bir built-in fonksiyon çağrısında (derivative builtins, `textureSample`, `textureSampleBias`, `textureSampleCompare`) uniformity analysis uniform control flow kanıtlayamıyorsa tetiklenir. Bkz. [§15.2 Uniformity](06-paralel-calisma-ve-dogruluk.md). |
| `subgroup_uniformity` | `error` | Subgroup veya quad built-in fonksiyon çağrısında uniformity analysis uniform control flow kanıtlayamıyorsa tetiklenir. Ayrıca `subgroupShuffleUp`/`Down`'da `delta` ve `subgroupShuffleXor`'da `mask` parametreleri uniform kanıtlanamıyorsa. |

- Tek token'lı tanınmayan triggering rule → warning tetiklemeli.
- Çoklu token formundaki tanınmayan triggering rule → diagnostic tetikleyebilir.
- Gelecek spec versiyonları bir kuralı kaldırabilir veya default severity'sini zayıflatabilir ve bu geriye uyumlu kabul edilir.

#### 2.3.3 Diagnostic Filtering

Filtrelenebilir triggering rule'a sahip bir diagnostic tetiklendiğinde, WGSL onu atma veya severity'sini modifiye etme mekanizmaları sağlar.

Bir **diagnostic filter** *DF* üç parametreye sahiptir:
- *AR*: Kaynak metnindeki **affected range**
- *NS*: Yeni **severity**
- *TR*: Bir **triggering rule**

*DF(AR, NS, TR)*'nin bir diagnostic *D*'ye uygulanması:
- *D*'nin triggering location'ı *AR* içindeyse ve triggering rule'u *TR* ise → *D*'nin severity'si *NS* olarak ayarlanır.
- Aksi halde → *D* değişmez.

**Range diagnostic filter**, `@diagnostic` attribute olarak belirli kaynak aralığının başında belirtilir:

| Yerleşim | Affected Range |
|----------|----------------|
| Compound statement başında | Compound statement |
| Function declaration başında | Function declaration |
| `if` statement başında | `if` + tüm `else if` ve `else` clause'ları |
| `switch` statement başında | Selector expression + switch_body |
| `switch_body` başında | switch_body |
| `loop` statement başında | Loop statement |
| `while` statement başında | Condition + loop body |
| `for` statement başında | for_header + loop body |
| Loop body'nin açılış brace'inden hemen önce | Loop body |
| `continuing_compound_statement` başında | continuing_compound_statement |

```wgsl
// Range diagnostic filter örneği
var<private> d: f32;
fn helper() -> vec4<f32> {
  // "if" gövdesinde derivative_uniformity diagnostic'ini devre dışı bırak.
  if (d < 0.5) @diagnostic(off, derivative_uniformity) {
    return textureSample(t, s, vec2(0, 0));
  }
  return vec4(0.0);
}
```

**Global diagnostic filter** ile tüm WGSL modülüne diagnostic filter uygulanabilir:

```wgsl
diagnostic(off, derivative_uniformity);
var<private> d: f32;
fn helper() -> vec4<f32> {
  if (d < 0.5) {
    // derivative_uniformity diagnostic'i global filter tarafından devre dışı.
    return textureSample(t, s, vec2(0, 0));
  } else {
    // derivative_uniformity diagnostic'i 'warning' severity'e ayarlandı.
    @diagnostic(warning, derivative_uniformity) {
      return textureSample(t, s, vec2(0, 0));
    }
  }
  return vec4(0.0);
}
```

**Çakışma kuralları:** İki diagnostic filter *DF(AR1, NS1, TR1)* ve *DF(AR2, NS2, TR2)*, (AR1 = AR2) ve (TR1 = TR2) ve (NS1 ≠ NS2) ise **çakışır**. Diagnostic filter'lar çakışmamalıdır.

WGSL'in diagnostic filter'ları, affected range'ler mükemmel şekilde iç içe geçecek şekilde tasarlanmıştır. DF1'in affected range'i DF2'ninki ile örtüşüyorsa, biri diğerinin tamamen içindedir.

### 2.4 Limits

WGSL implementasyonu aşağıdaki limitleri karşılayan shader'ları destekler. Belirtilen limitlerin ötesindeki shader'lar da desteklenebilir.

> **Not:** Implementasyon, belirtilen limitlerin ötesindeki bir shader'ı desteklemiyorsa hata vermeli.

| Limit | Minimum Desteklenen Değer |
|-------|--------------------------|
| Structure type'ta maksimum üye sayısı | 1023 |
| Composite type'ın maksimum nesting depth'i | 15 |
| Fonksiyonda brace-enclosed statement'ların maksimum nesting depth'i | 127 |
| Fonksiyon için maksimum parametre sayısı | 255 |
| Switch statement'ta maksimum case selector değer sayısı (default clause dahil) | 1023 |
| Bir shader tarafından statically accessed edilen `private` address space değişkenlerinin maksimum toplam byte-size'ı | 8192 |
| Bir fonksiyonda bildirilen `function` address space değişkenlerinin maksimum toplam byte-size'ı | 8192 |
| Bir shader tarafından statically accessed edilen `workgroup` address space değişkenlerinin maksimum toplam byte-size'ı | 16384 |
| Array type value constructor expression'da maksimum eleman sayısı | 2047 |

---

## §3 Textual Structure

### 3.1 Parsing

WGSL modülünü ayrıştırmak (parse etmek) için:

1. **Comment kaldırma:** İlk comment'i boşluk (`U+0020`) ile değiştir. Comment kalmayıncaya kadar tekrarla.
2. **Template list keşfi:** [§3.9 Template Lists](#39-template-lists)'teki algoritmayı kullanarak `<` ve `>` karakterlerinin template list sınırlayıcı mı yoksa karşılaştırma operatörü mü olduğunu belirle.
3. **Grammar ile eşleştirme:** Tüm metni `translation_unit` grammar kuralıyla eşleştirmeye çalış. Parser, **LALR(1)** (bir token lookahead) kullanır ve aşağıdaki özelleştirme ile çalışır:
   - Tokenizasyon parsing ile interleaved çalışır ve context-aware'dir.
   - Parser sonraki token'ı talep ettiğinde:
     - Önce blankspace atlanır.
     - Sonraki code point bir template list başlangıcıysa → `_template_args_start` token'ı döndürülür.
     - Template list bitişiyse → `_template_args_end` token'ı döndürülür.
     - Aksi halde: **token candidate** (kalan unconsumed code point'lerin boş olmayan prefix'inden oluşan geçerli WGSL token), en uzun geçerli lookahead token seçilir.

**Shader-creation error** oluşur eğer:
- Kaynak metin tamamen geçerli token dizisine dönüştürülemiyorsa
- `translation_unit` grammar kuralı tüm token dizisini eşleştirmiyorsa

> **Not:** Yukarıdaki prosedür template list discovery'yi ayrı bir aşama (adım 2) olarak açıklar. Alternatif olarak, template list discovery tokenizasyon ile interleaved çalıştırılabilir. Bu yaklaşımda, `_disambiguate_template` sentetik token'ı grammar kuralında template list olabilecek her yere yerleştirilir.

### 3.2 Blankspace and Line Breaks

**Blankspace**, Unicode `Pattern_White_Space` özelliğinden bir veya daha fazla code point'in birleşimidir:

| Code Point | Karakter |
|-----------|----------|
| `U+0020` | space (boşluk) |
| `U+0009` | horizontal tab |
| `U+000A` | line feed |
| `U+000B` | vertical tab |
| `U+000C` | form feed |
| `U+000D` | carriage return |
| `U+0085` | next line |
| `U+200E` | left-to-right mark |
| `U+200F` | right-to-left mark |
| `U+2028` | line separator |
| `U+2029` | paragraph separator |

**Line break** (satır sonu), satırın sonunu gösteren ardışık blankspace code point dizisidir. UAX14 Section 6.1 (LB4s LB5) kurallarıyla tanımlanır:

- Line feed (`U+000A`)
- Vertical tab (`U+000B`)
- Form feed (`U+000C`)
- Carriage return (`U+000D`) — ardından line feed **gelmediğinde**
- Carriage return (`U+000D`) ardından line feed (`U+000A`)
- Next line (`U+0085`)
- Line separator (`U+2028`)
- Paragraph separator (`U+2029`)

> **Not:** Kaynak metin konumlarını satır numarası cinsinden raporlayan diagnostic'ler, satırları saymak için line break tanımını kullanmalıdır.

### 3.3 Comments

**Comment** (yorum), WGSL programının geçerliliğini veya anlamını etkilemeyen metin aralığıdır; ancak token'ları ayırabilir. İki türü vardır:

**Line-ending comment** — `//` (iki `U+002F`) ile başlar, sonraki line break'e veya programın sonuna kadar devam eder:

```wgsl
const f = 1.5; // Bu bir satır sonu yorumudur.
```

**Block comment** — `/*` ile başlar, `*/` ile biter. **İç içe yerleştirilebilir** (nested):

```wgsl
const g = 2.5; /* Bu bir blok yorumudur
                   ve birden fazla satıra yayılır.
                   /* Blok yorumlar iç içe olabilir. */
                   Ancak tüm blok yorumlar sonlandırılmalıdır.
                */
```

> **Not:** Blok yorumlar iç içe yerleştirilebildiği için düzenli ifade (regex) ile tanınamaz. Bu, Regular Languages için Pumping Lemma'nın bir sonucudur.

### 3.4 Tokens

**Token**, ardışık code point'lerden oluşan aşağıdaki türlerden birini oluşturan dizidir:

- **Literal** — Bir değer temsili
- **Keyword** — Önceden tanımlanmış bir dil kavramına referans
- **Reserved word** — Gelecek kullanım için ayrılmış kelime
- **Syntactic token** — Operatör veya noktalama işareti
- **Identifier** — Bir isim olarak kullanılan token
- **Context-dependent name** — Yalnızca belirli gramatikal bağlamlarda kullanılan isim

### 3.5 Literals

**Literal** (değişmez değer) aşağıdakilerden biridir:

- **Boolean literal** — `true` veya `false`
- **Numeric literal** — Integer literal veya floating point literal

```bnf
literal :
  | int_literal
  | float_literal
  | bool_literal
```

#### 3.5.1 Boolean Literals

```wgsl
const a = true;
const b = false;
```

```bnf
bool_literal :
  | 'true'
  | 'false'
```

#### 3.5.2 Numeric Literals

Numeric literal'ın formu pattern-matching ile tanımlanır.

**Integer literal** şu formda olabilir:
- `0` (yalın sıfır)
- İlk rakam 0 olmayan ondalık basamak dizisi (ör. `123`)
- `0x` veya `0X` prefix'li onaltılık basamak dizisi (ör. `0x3f`)
- Ardından opsiyonel `i` (i32 belirtir) veya `u` (u32 belirtir) suffix'i

```bnf
decimal_int_literal :
  | /0[iu]?/
  | /[1-9][0-9]*[iu]?/

hex_int_literal :
  | /0[xX][0-9a-fA-F]+[iu]?/
```

```wgsl
const a = 1u;    // u32 tipi
const b = 123;   // AbstractInt tipi
const c = 0;     // AbstractInt tipi
const d = 0i;    // i32 tipi
const e = 0x123; // AbstractInt hex
const f = 0X3fu; // u32 hex
```

> **Not:** Sıfır olmayan bir integer literal'da başta sıfır bulunması (ör. `012`) yasaktır — diğer dillerdeki "başta sıfır = octal" gösterimiyle karışıklığı önlemek için.

**Floating point literal** — decimal veya hexadecimal formda:

**Decimal floating point literal:**
- Rakam dizisinden oluşan bir significand (aralarında opsiyonel `.` ondalık nokta)
- Ardından opsiyonel `e`/`E` üs bölümü (opsiyonel `+`/`-` işareti ile)
- Ardından opsiyonel `f` (f32) veya `h` (f16) suffix'i
- `.`, üs veya `f`/`h` suffix'inden en az biri mevcut **olmalıdır** (yoksa token integer literal'dır)

```bnf
decimal_float_literal :
  | /0[fh]/
  | /[1-9][0-9]*[fh]/
  | /[0-9]*\.[0-9]+([eE][+-]?[0-9]+)?[fh]?/
  | /[0-9]+\.[0-9]*([eE][+-]?[0-9]+)?[fh]?/
  | /[0-9]+[eE][+-]?[0-9]+[fh]?/
```

```wgsl
const a = 0.e+4f;  // f32
const b = 01.;     // AbstractFloat
const c = .01;     // AbstractFloat
const d = 12.34;   // AbstractFloat
const f = .0f;     // f32
const g = 0h;      // f16
const h = 1e-3;    // AbstractFloat
```

**Decimal floating point literal'ın matematiksel değeri:**
- Significand 20 veya daha az anlamlı rakama sahipse → doğrudan kullanılır.
- 20'den fazla anlamlı rakam varsa → 20. anlamlı rakamdan sonraki tüm rakamlar 0 yapılır (truncated) veya 20. rakam 1 artırılır (truncated_next). Bu bir implementasyon seçimidir.
- Matematiksel değer: *effective_significand* × 10^(üs). Üs belirtilmediyse 0 varsayılır.

> **Not:** Decimal significand 20 decimal basamaktan sonra kesilerek yaklaşık log(10)/log(2)×20 ≈ 66.4 anlamlı bit korunur.

**Hexadecimal floating point literal:**
- `0x` veya `0X` prefix'i
- Hex basamaklardan oluşan significand (opsiyonel `.` hex nokta ile)
- Opsiyonel `p`/`P` üs bölümü (decimal sayı, opsiyonel işaret)
- Opsiyonel `f`/`h` suffix'i
- `.` veya üs'ten en az biri mevcut **olmalıdır**

```bnf
hex_float_literal :
  | /0[xX][0-9a-fA-F]*\.[0-9a-fA-F]+([pP][+-]?[0-9]+[fh]?)?/
  | /0[xX][0-9a-fA-F]+\.[0-9a-fA-F]*([pP][+-]?[0-9]+[fh]?)?/
  | /0[xX][0-9a-fA-F]+[pP][+-]?[0-9]+[fh]?/
```

```wgsl
const a = 0xa.fp+2;  // AbstractFloat
const b = 0x1P+4f;   // f32
const c = 0X.3;      // AbstractFloat
const d = 0x3p+2h;   // f16
const e = 0X1.fp-4;  // AbstractFloat
const f = 0x3.2p+2h; // f16
```

**Hexadecimal floating point literal'ın matematiksel değeri:**
- Significand 16 hex basamakta kesilir (yaklaşık 4×16 = 64 anlamlı bit).
- Matematiksel değer: *effective_significand* × 2^(üs).

**Suffix ile tip eşleme:**

| Literal Türü | Suffix | Tip | Örnek |
|-------------|--------|-----|-------|
| Integer literal | `i` | `i32` | `42i` |
| Integer literal | `u` | `u32` | `42u` |
| Integer literal | *(yok)* | `AbstractInt` | `124` |
| Floating point literal | `f` | `f32` | `42f`, `1e5f`, `0x1.0p10f` |
| Floating point literal | `h` | `f16` | `42h`, `1e5h`, `0x1.0p10h` |
| Floating point literal | *(yok)* | `AbstractFloat` | `1e5`, `1.2`, `0x1.0p10` |

**Shader-creation error** oluşur eğer:
- `i`/`u` suffix'li integer literal hedef tipte temsil edilemiyorsa
- `f`/`h` suffix'li hex float literal overflow yapıyorsa veya hedef tipte tam temsil edilemiyorsa
- `f`/`h` suffix'li decimal float literal overflow yapıyorsa
- `h` suffix'li literal kullanılırken `f16` extension etkin değilse

> **Not:** `0x1.00000001p0` hex float değeri 33 significand bit gerektirir; `f32` yalnızca 23 explicit significand bit'e sahiptir.

> **Not:** Hex float literal'ı `f` suffix ile `f32` tipine zorlamak için binary üs de kullanmanız gerekir. Örneğin `0x1p0f` yazın. Karşılaştırma: `0x1f` bir hex integer literal'dır.

### 3.6 Keywords

**Keyword** (anahtar kelime), önceden tanımlanmış bir dil kavramına atıfta bulunan token'dır.

Tüm keyword'lerin listesi için [§16.1 Keyword Summary](#161-keyword-summary) bölümüne bakınız.

### 3.7 Identifiers

**Identifier** (tanımlayıcı), isim olarak kullanılan bir token türüdür. Bkz. [§5 Declaration and Scope](#5-declaration-and-scope).

WGSL iki grammar nonterminal'i kullanır:
- **`ident`** — Bildirilen bir nesneyi isimlendirmek için kullanılır.
- **`member_ident`** — Bir structure tipinin üyesini isimlendirmek için kullanılır.

```bnf
ident :
  | ident_pattern_token _disambiguate_template

member_ident :
  | ident_pattern_token
```

Identifier'ın formu, Unicode Standard Annex #31 (Unicode 14.0.0) temel alınarak belirlenir:

```
<Identifier> := <Start> <Continue>* (<Medial> <Continue>+)*
<Start>      := XID_Start + U+005F
<Continue>   := <Start> + XID_Continue
<Medial>     :=
```

Bu, ASCII olmayan code point'li identifier'ların geçerli olduğu anlamına gelir: `Δέλτα`, `réflexion`, `Кызыл`, `𐰓𐰏𐰇`, `朝焼け`, `سلام`, `검정`, `שָׁלוֹם`, `गुलाबी`, `փիրուզ`.

**Kısıtlamalar:**
- Identifier, bir keyword veya reserved word ile aynı yazılışta **olmamalıdır**.
- Identifier, tek alt çizgi `_` (`U+005F`) **olmamalıdır**.
- Identifier, `__` (iki alt çizgi) ile **başlamamalıdır**.

```bnf
ident_pattern_token :
  | /([_\p{XID_Start}][\p{XID_Continue}]+)|([\p{XID_Start}])/u
```

> **Not:** Bazı built-in fonksiyonların dönüş tipi, adı WGSL kaynağında kullanılamayan structure tiplerdir. Bu tipler, `__` ile başlayan isimlere sahipmiş gibi tanımlanır. Sonuç değeri `let` veya `var` ile tip çıkarımı kullanılarak saklanabilir. `frexp` ve `modf` kullanım örneklerine bakınız.

#### 3.7.1 Identifier Comparison

İki WGSL identifier'ı ancak ve ancak aynı code point dizisinden oluşuyorsa **aynıdır**.

> **Not:** Bu spesifikasyon, karşılaştırma amacıyla Unicode normalizasyonuna izin vermez. Görsel ve anlamsal olarak aynı olup farklı Unicode karakter dizileri kullanan değerler eşleşmeyecektir.

> **Not:** Kullanıcı ajanı, bir identifier'ın tüm örnekleri bir homograf ile değiştirildiğinde modülün anlamı değişecekse geliştirici görünür uyarıları yayınlamalıdır.

### 3.8 Context-Dependent Names

**Context-dependent name**, bir kavramı isimlendirmek için kullanılan ancak yalnızca belirli gramatikal bağlamlarda geçerli olan token'dır. Token'ın yazılışı bir identifier ile aynı olabilir, ancak bildirilen bir nesneye resolve olmaz. Token bir keyword veya reserved word **olmamalıdır**.

#### 3.8.1 Attribute Names

Bkz. [§12 Attributes](05-gpu-arayuzu-ve-bellek.md#12-attributes).

Attribute isimleri:

`align`, `binding`, `builtin`, `compute`, `const`, `diagnostic`, `fragment`, `group`, `id`, `interpolate`, `invariant`, `location`, `blend_src`, `must_use`, `size`, `vertex`, `workgroup_size`

#### 3.8.2 Built-in Value Names

Built-in value name-token, bir built-in value'nun adında kullanılan token'dır.

Bkz. [§13.3.1.1 Built-in Inputs and Outputs](05-gpu-arayuzu-ve-bellek.md#13311-built-in-inputs-and-outputs).

```bnf
builtin_value_name :
  | ident_pattern_token
```

Built-in value isimleri:

`vertex_index`, `instance_index`, `position`, `front_facing`, `frag_depth`, `sample_index`, `sample_mask`, `local_invocation_id`, `local_invocation_index`, `global_invocation_id`, `workgroup_id`, `num_workgroups`, `subgroup_invocation_id`, `subgroup_size`, `primitive_index`, `subgroup_id`, `num_subgroups`

#### 3.8.3 Diagnostic Rule Names

Diagnostic name-token, bir diagnostic triggering rule'un adında kullanılan token'dır.

Bkz. [§2.3 Diagnostics](#23-diagnostics).

```bnf
diagnostic_name_token :
  | ident_pattern_token
```

Önceden tanımlı diagnostic rule isimleri: `derivative_uniformity`, `subgroup_uniformity`

#### 3.8.4 Diagnostic Severity Control Names

Diagnostic filter severity kontrol isimleri [§2.3 Diagnostics](#23-diagnostics)'te listelenmiştir ve identifier ile aynı forma sahiptir:

```bnf
severity_control_name :
  | ident_pattern_token
```

Severity kontrol isimleri: `error`, `warning`, `info`, `off`

#### 3.8.5 Extension Names

**Enable-extension** isimleri [§4.1.1 Enable Extensions](#411-enable-extensions)'da listelenmiştir:

```bnf
enable_extension_name :
  | ident_pattern_token
```

Enable-extension isimleri: `f16`, `clip_distances`, `dual_source_blending`, `subgroups`, `primitive_index`

**Language extension** isimleri [§4.1.2 Language Extensions](#412-language-extensions)'da listelenmiştir:

```bnf
language_extension_name :
  | ident_pattern_token
```

Language extension isimleri: `readonly_and_readwrite_storage_textures`, `packed_4x8_integer_dot_product`, `unrestricted_pointer_parameters`, `pointer_composite_access`, `uniform_buffer_standard_layout`, `subgroup_id`, `subgroup_uniformity`, `texture_and_sampler_let`, `texture_formats_tier1`

#### 3.8.6 Interpolation Type Names

Interpolation type name-token, bir interpolation type'ın adında kullanılır.

Bkz. [§13.3.1.4 Interpolation](05-gpu-arayuzu-ve-bellek.md#13314-interpolation).

Interpolation type isimleri: `perspective`, `linear`, `flat`

#### 3.8.7 Interpolation Sampling Names

Interpolation sampling name-token, bir interpolation sampling'in adında kullanılır.

```bnf
interpolate_sampling_name :
  | ident_pattern_token
```

Interpolation sampling isimleri: `center`, `centroid`, `sample`, `first`, `either`

#### 3.8.8 Swizzle Names

Swizzle isimleri, [vector access expression'larında](03-degiskenler-ve-ifadeler.md#vector-access-expressions) kullanılır:

```bnf
swizzle_name :
  | /[rgba]/
  | /[rgba][rgba]/
  | /[rgba][rgba][rgba]/
  | /[rgba][rgba][rgba][rgba]/
  | /[xyzw]/
  | /[xyzw][xyzw]/
  | /[xyzw][xyzw][xyzw]/
  | /[xyzw][xyzw][xyzw][xyzw]/
```

İki swizzle takımı vardır: **rgba** (renk bileşenleri) ve **xyzw** (koordinat bileşenleri). Bir swizzle ifadesinde her iki takım karıştırılamaz.

### 3.9 Template Lists

**Template parameterization**, genel bir kavramı modifiye eden parametreler belirtme yoludur. Genel kavramın ardına bir **template list** eklenerek yazılır.

Blankspace ve comment'ler göz ardı edilerek, bir **template list** şunlardan oluşur:
- `<` (`U+003C`) başlangıç code point'i
- Virgülle ayrılmış bir veya daha fazla **template parameter**
- Opsiyonel sonda virgül
- `>` (`U+003E`) bitiş code point'i

> **Not:** Örneğin, `vec3<f32>` ifadesinde `vec3` genel kavramdır ve `<f32>` bir template list'tir. Birlikte belirli bir vector tipini belirtirler.

> **Not:** `var<storage,read_write>` ifadesi `var` kavramını `storage` ve `read_write` template parametreleriyle modifiye eder.

> **Not:** `array<vec4<f32>>` ifadesinde iki template parameterizasyon vardır: `vec4<f32>` ve `array<vec4<f32>>`.

`<` ve `>` code point'leri ayrıca şu bağlamlarda kullanılır:
- `relational_expression` içinde karşılaştırma operatörü olarak
- `shift_expression` içinde shift operatörü olarak (<< ve >>)
- Shift + assignment compound operatörü olarak

Sözdizimsel belirsizlik **template list lehine** çözülür: template list'ler parsing'in erken bir aşamasında, declaration/expression/statement parse edilmeden önce keşfedilir.

#### Template List Discovery Algoritması

**Input:** Program kaynak metni

**Record türleri:**
- `UnclosedCandidate` = { `position`: kaynak metni konumu, `depth`: expression nesting derinliği }
- `TemplateList` = { `start_position`: `<` code point konumu, `end_position`: `>` code point konumu }

**Output:** `DiscoveredTemplateLists` — `TemplateList` kayıtlarının listesi

**Prosedür:**
1. `DiscoveredTemplateLists` boş bir liste olarak başlat
2. `Pending` boş bir `UnclosedCandidate` stack'i olarak başlat
3. `CurrentPosition` = 0 (metin sonu geldiğinde algoritma sonlanır)
4. `NestingDepth` = 0
5. Tekrarla:
   - `CurrentPosition`'ı blankspace, comment ve literal'lar üzerinden ilerlet
   - `ident_pattern_token` eşleşiyorsa → ilerlet, sonra:
     - `<` varsa → `UnclosedCandidate(position=CurrentPosition, depth=NestingDepth)` push et
       - Ardından `<` gelirse → `<<` shift operatörüdür, stack'ten pop et
       - Ardından `=` gelirse → `<=` karşılaştırmadır, stack'ten pop et
   - `>` code point'inde: `Pending` boş değilse ve top entry'nin depth'i NestingDepth'e eşitse → template list keşfedildi, `DiscoveredTemplateLists`'e ekle
     - Değilse → `>=` kontrol et, eğer öyleyse `=`'i atla
   - `(` veya `[` → `NestingDepth` + 1
   - `)` veya `]` → `Pending`'den uygun entry'leri pop et, `NestingDepth` - 1 (min 0)
   - `!` → `!=` kontrol et
   - `=` → `==` kontrol et; tek `=` ise assignment = stack'i temizle
   - `;`, `{`, `:` → expression'da olamaz, stack'i temizle ve `NestingDepth` = 0
   - `&&`, `||` → nesting'i korur ama expression'lar arasındadır, benzer şekilde işlenir
   - Diğer code point'leri → ilerlet

---

## §4 Directives

Directive'ler, tüm modül için geçerli olan modül düzeyinde modifikasyonları belirtir.

```bnf
global_directive :
  | diagnostic_directive
  | enable_directive
  | requires_directive
```

### 4.1 Extensions

WGSL, temel dil özelliklerinin ötesinde ek yetenekler sunan iki extension mekanizması sağlar.

#### 4.1.1 Enable Extensions

**Enable extension**, kullanmadan önce etkinleştirilmesi gereken bir WGSL özelliğidir. Bir `enable` directive'i, modül için belirtilen extension'ları etkinleştirir.

```bnf
enable_directive :
  | 'enable' enable_extension_list ';'

enable_extension_list :
  | enable_extension_name (',' enable_extension_name)* ','?
```

```wgsl
enable f16;
enable subgroups;
```

Tanımlı enable extension'lar:

| Extension | Açıklama |
|-----------|----------|
| `f16` | 16-bit floating point tipin (`f16`) kullanımını etkinleştirir. `h` suffix'li literal'lar da bu extension'ı gerektirir. |
| `clip_distances` | `clip_distances` built-in output'unu etkinleştirir. |
| `dual_source_blending` | `blend_src` attribute ile dual-source blending'i etkinleştirir. |
| `subgroups` | Subgroup built-in fonksiyonlarını etkinleştirir (subgroup vote, arithmetic, ballot vb.). `subgroup_invocation_id` ve `subgroup_size` built-in input'larını sağlar. |
| `primitive_index` | `primitive_index` built-in input'unu etkinleştirir; triangle strip primitif'i içindeki üçgen indeksini açığa çıkarır. |

> **Not:** Bir extension disabled olduğunda, o extension'a ait built-in fonksiyonlar, tipler veya değerler kullanılamaz.

Kurallar:
- Tanınmayan extension ismi → shader-creation error
- Aynı extension birden fazla kez etkinleştirilebilir
- Extension'lar tüm modül boyunca geçerlidir (sadece sonrasında değil)

#### 4.1.2 Language Extensions

**Language extension**, WGSL'in taşınabilir davranışını genişleten bir özelliktir. `requires` directive'i ile bildirilir.

```bnf
requires_directive :
  | 'requires' language_extension_list ';'

language_extension_list :
  | language_extension_name (',' language_extension_name)* ','?
```

```wgsl
requires readonly_and_readwrite_storage_textures;
```

Tanımlı language extension'lar:

| Extension | Açıklama |
|-----------|----------|
| `readonly_and_readwrite_storage_textures` | `read` ve `read_write` access mode'lu storage texture'ları etkinleştirir. |
| `packed_4x8_integer_dot_product` | `packed4x8IntegerDotProduct` abstract tipini ve ilgili built-in fonksiyonları etkinleştirir. |
| `unrestricted_pointer_parameters` | `storage`, `uniform` ve `workgroup` adres space'lerine pointer parametreleri geçirilmesini sağlar. |
| `pointer_composite_access` | Pointer tipi üzerinde composite access (array index, member access) ifadelerine izin verir. |
| `uniform_buffer_standard_layout` | Uniform buffer'lardaki structure'ların standard layout kurallarına uymasını sağlar. |
| `subgroup_id` | Subgroup ve workgroup içinde subgroup tanımlayıcı built-in'leri sağlar. |
| `subgroup_uniformity` | Subgroup built-in fonksiyonları için uniformity analiz kuralını etkinleştirir. |
| `texture_and_sampler_let` | `texture` ve `sampler` handle'larını `let` bildirimleriyle saklamayı sağlar. |
| `texture_formats_tier1` | Tier 1 texture format'larını (ör. `r8unorm`, `rg8unorm` vb.) storage texture'lar için etkinleştirir. |

Kurallar:
- Tanınmayan language extension ismi → shader-creation error
- Desteklenmeyen extension → pipeline-creation error (bilinen ama bu implementasyon tarafından desteklenmeyen extension'lar)

### 4.2 Global Diagnostic Filter

**Global diagnostic filter**, tüm WGSL modülünü kapsayan bir diagnostic filter uygular.

```bnf
diagnostic_directive :
  | 'diagnostic' diagnostic_control ';'

diagnostic_control :
  | '(' severity_control_name ',' diagnostic_rule_name ')'
```

```wgsl
// derivative_uniformity diagnostic'ini tüm modülde devre dışı bırak
diagnostic(off, derivative_uniformity);

// subgroup_uniformity'yi 'warning' seviyesine düşür
diagnostic(warning, subgroup_uniformity);
```

> **Not:** Birden fazla global diagnostic filter bildirilebilir. Aynı triggering rule için birden fazla <u>çakışan</u> global diagnostic filter bildirmek hata üretir. Bkz. [§2.3.3 Diagnostic Filtering](#233-diagnostic-filtering).

---

## §5 Declaration and Scope

**Declaration** (bildirim), bir nesneye isim veren yapıdır. WGSL'de bildirmeler şunlardır:

- **Type declarations**: `struct`, `alias`
- **Value declarations**: `const`, `let`, `override`
- **Variable declarations**: `var`
- **Function declarations**: `fn`
- **Formal parameter declarations**: fonksiyon parametre listesindeki bildirimler

```bnf
global_value_decl :
  | 'const' optionally_typed_ident '=' expression
  | 'override' optionally_typed_ident ('=' expression)?

global_variable_decl :
  | variable_decl ('=' expression)?

type_alias_decl :
  | 'alias' ident '=' type_specifier
```

### 5.1 Scope

Bildirimler bir **scope** (kapsam) içindedir. WGSL'de scope'lar iç içe geçer. Bir isim, bildirildiği scope'ta ve onun tüm alt scope'larında erişilebilir.

#### Module scope

Module scope, tüm modülün scope'udur. Module scope'ta şunlar bildirilebilir:
- Global `var` değişkenleri
- `const`, `override`, `alias` bildirimleri
- `struct` ve `fn` bildirimleri

Module scope bildirimlerin sırası önemli değildir — bir bildirim, kendisinden önce veya sonra bildirilen başka bir nesneye referans verebilir (forward reference).

#### Function scope

Function scope, bir fonksiyon bildiriminin scope'udur. Fonksiyon parametreleri bu scope'ta bildirilir.

#### Compound statement scope

Compound statement (`{ }`) yeni bir scope oluşturur. `let`, `var` ve `const` bu scope içinde bildirilebilir.

### 5.2 Name Resolution

Bir `ident`'in kullanımı, bir **declaration**'a **resolve** eder. Resolution kuralları:

1. **Enclosing scope:** İsim, onu kapsayan en yakın scope'taki bildirime resolve edilir.
2. **Shadowing yasaktır:** Bir iç scope'taki bildirim, dış scope'taki bir bildirimi gölgeleyemez — bu bir **shader-creation error**'dır.
3. **Forward reference (module scope):** Module scope'ta bildirimler birbirlerine bildiri sıralarından bağımsız olarak referans verebilir.
4. **Forward reference (function scope):** Fonksiyon gövdesi içindeki bildirimler forward reference yapamaz — kullanımdan önce bildirilmelidir.

```wgsl
// Module scope — sıra önemli değildir
fn foo() -> i32 { return bar(); }
fn bar() -> i32 { return 42; }

// Shadowing YASAKTIR
var<private> x: i32;
fn example() {
  // var x: i32;  // HATA: dış scope'taki 'x'i gölgeler
  let y = x;     // OK: module scope'taki 'x'e resolve eder
}
```

### 5.3 Predeclared Objects

WGSL, birçok **predeclared** (önceden bildirilmiş) nesneye sahiptir. Bunlar module scope başlamadan önce mevcut olan örtük bir scope'ta bulunur:

- **Tipler:** `bool`, `i32`, `u32`, `f32`, `f16`, `vec2`, `vec3`, `vec4`, `mat2x2`, `mat3x3`, `mat4x4`, `array`, `atomic`, `ptr`, `sampler`, `sampler_comparison`, texture tipleri, vb.
- **Built-in fonksiyonlar:** `abs`, `sin`, `cos`, `dot`, `cross`, `normalize`, vb. (tam liste [§17'de](07-built-in-kutuphanesi.md))
- **Type generators:** `vec2`, `vec3`, `vec4`, `mat2x2`, vb. hem değer oluşturucu (value constructor) hem de tip olarak kullanılabilir.

Bir identifier, predeclared bir isimle aynı isme sahip olabilir — bu durumda predeclared nesne gölgelenir ve bu scope'ta erişilemez hale gelir.

> **Not:** Predeclared nesneyi gölgelemek geçerlidir, ancak kafa karışıklığına yol açabileceğinden genellikle önerilmez.

---

### 16.1 Keyword Summary

Aşağıdaki token'lar WGSL **keyword**'leridir:

| Keyword | Açıklama |
|---------|----------|
| `alias` | Tip takma adı bildirimi |
| `break` | Döngü veya switch'ten çıkış |
| `case` | Switch case clause |
| `const` | Sabit bildirim |
| `const_assert` | Derleme zamanı assert |
| `continue` | Döngüde sonraki iterasyona geçiş |
| `continuing` | Loop continuing bloğu |
| `default` | Switch default clause |
| `diagnostic` | Diagnostic filter |
| `discard` | Fragment shader'da fragment'ı at |
| `else` | Koşullu dallanma — aksi durum |
| `enable` | Extension etkinleştirme |
| `false` | Boolean false değeri |
| `fn` | Fonksiyon bildirimi |
| `for` | For döngüsü |
| `if` | Koşullu dallanma |
| `let` | Sabit değer bildirimi (runtime) |
| `loop` | Genel döngü |
| `override` | Pipeline-sabit bildirim |
| `requires` | Language extension gerekliliği |
| `return` | Fonksiyondan dönüş |
| `struct` | Structure tipi bildirimi |
| `switch` | Switch statement |
| `true` | Boolean true değeri |
| `var` | Değişken bildirimi |
| `while` | While döngüsü |

### 16.2 Reserved Words

Aşağıdaki kelimeler gelecekte kullanılmak üzere **ayrılmıştır** ve identifier olarak kullanılamaz:

`NULL`, `Self`, `abstract`, `active`, `alignas`, `alignof`, `as`, `asm`, `asm_fragment`, `async`, `attribute`, `auto`, `await`, `become`, `binding_array`, `cast`, `catch`, `class`, `co_await`, `co_return`, `co_yield`, `coherent`, `column_major`, `common`, `compile`, `compile_fragment`, `concept`, `const_cast`, `consteval`, `constexpr`, `constinit`, `crate`, `debugger`, `decltype`, `delete`, `demote`, `demote_to_helper`, `do`, `dynamic_cast`, `enum`, `explicit`, `export`, `extends`, `extern`, `external`, `fallthrough`, `filter`, `final`, `finally`, `friend`, `from`, `fxgroup`, `get`, `goto`, `groupshared`, `highp`, `impl`, `implements`, `import`, `in`, `inline`, `instanceof`, `interface`, `layout`, `lowp`, `macro`, `macro_rules`, `match`, `mediump`, `meta`, `mod`, `module`, `move`, `mut`, `mutable`, `namespace`, `new`, `nil`, `noexcept`, `noinline`, `nointerpolation`, `noperspective`, `null`, `nullptr`, `of`, `operator`, `package`, `packoffset`, `partition`, `pass`, `patch`, `pixelfragment`, `precise`, `precision`, `premerge`, `private`, `protected`, `pub`, `public`, `readonly`, `ref`, `regardless`, `register`, `reinterpret_cast`, `require`, `resource`, `restrict`, `self`, `set`, `shared`, `sizeof`, `smooth`, `snorm`, `static`, `static_assert`, `static_cast`, `std`, `subroutine`, `super`, `target`, `template`, `this`, `thread_local`, `throw`, `trait`, `try`, `type`, `typedef`, `typeid`, `typename`, `typeof`, `union`, `unless`, `unorm`, `unsafe`, `unsized`, `use`, `using`, `varying`, `virtual`, `volatile`, `wgsl`, `where`, `with`, `writeonly`, `yield`

### 16.3 Syntactic Tokens

Aşağıdaki operatör ve noktalama (punctuation) simgeleri WGSL **syntactic token**'larıdır:

| Token | Açıklama |
|-------|----------|
| `&` | Bitwise AND / address-of |
| `&&` | Logical AND (short-circuit) |
| `->` | Fonksiyon dönüş tipi belirteci |
| `@` | Attribute marker |
| `/` | Bölme |
| `!` | Logical NOT |
| `[` `]` | Array indeks erişimi |
| `{` `}` | Compound statement / struct body |
| `(` `)` | Gruplama / fonksiyon çağrısı |
| `:` | Tip annotation ayracı |
| `,` | Parametre/eleman ayracı |
| `;` | Statement sonlandırıcı |
| `=` | Atama |
| `==` | Eşitlik karşılaştırma |
| `!=` | Eşitsizlik karşılaştırma |
| `<` | Küçüktür / template list başlangıcı |
| `<=` | Küçük eşit |
| `>` | Büyüktür / template list bitişi |
| `>=` | Büyük eşit |
| `<<` | Sola shift |
| `>>` | Sağa shift |
| `+` | Toplama |
| `-` | Çıkarma / negatif |
| `*` | Çarpma / pointer dereference |
| `%` | Modulo (kalan) |
| `^` | Bitwise XOR |
| `~` | Bitwise NOT (complement) |
| `\|` | Bitwise OR |
| `\|\|` | Logical OR (short-circuit) |
| `+=` | Toplama + atama |
| `-=` | Çıkarma + atama |
| `*=` | Çarpma + atama |
| `/=` | Bölme + atama |
| `%=` | Modulo + atama |
| `&=` | Bitwise AND + atama |
| `\|=` | Bitwise OR + atama |
| `^=` | Bitwise XOR + atama |
| `<<=` | Sola shift + atama |
| `>>=` | Sağa shift + atama |
| `++` | Increment |
| `--` | Decrement |
| `.` | Member erişimi |
| `_` | Phony assignment hedefi |

---

> **Sonraki:** [Tip Sistemi →](02-tip-sistemi.md)
