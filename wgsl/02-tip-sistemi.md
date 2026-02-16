---
title: Tip Sistemi (Types)
section: 6
source: "W3C WGSL Spec §6"
---

# 2. Tip Sistemi

> WGSL'in en kritik bölümü: type checking, plain types, memory views, texture/sampler ve type alias'lar.

---

## §6 Types

Programlar **değerler** hesaplar. WGSL'de bir **tip** (type), bir değerler kümesidir ve her değer tam olarak bir tipe aittir. Bir değerin tipi, o değer üzerinde gerçekleştirilebilecek operasyonların sözdizimini ve semantiğini belirler.

Örneğin, matematiksel `1` sayısı WGSL'de şu farklı değerlere karşılık gelir:

| Değer | Tip |
|-------|-----|
| `1i` | 32-bit signed integer (i32) |
| `1u` | 32-bit unsigned integer (u32) |
| `1.0f` | 32-bit floating point (f32) |
| `1.0h` | 16-bit floating point (f16, `enable f16` gerekir) |
| `1` | AbstractInt |
| `1.0` | AbstractFloat |

WGSL bunları farklı değerler olarak ele alır çünkü makine temsilleri ve operasyonları farklıdır.

Bir tip ya **predeclared** (önceden bildirilmiş) ya da WGSL kaynağında bir **declaration** ile oluşturulur.

Bazı tipler **template parameterization** olarak ifade edilir. Bir **type-generator**, template list ile parametrelendirildiğinde bir tip gösteren predeclared nesnedir. Örneğin `atomic<u32>`, `atomic` type-generator'ını `<u32>` template list'i ile birleştirir.

Bazı WGSL tipleri yalnızca kaynak program analizi ve çalışma zamanı davranışının belirlenmesi için kullanılır; WGSL kaynak metninde görünmez.

> **Not:** Reference type'lar WGSL modüllerinde yazılmaz. Bkz. [§6.4.3 Reference and Pointer Types](#643-reference-and-pointer-types).

---

### 6.1 Type Checking

Bir WGSL değeri bir **expression** (ifade) değerlendirilerek hesaplanır. Expression, WGSL grammar kurallarından adı "expression" ile biten biridir. Bir expression *E*, kendisi içinde yer alan **subexpression**'lar içerebilir.

Bir expression değerlendirmesinin ürettiği değer şunlara bağlıdır:
- **Static context**: İfadeyi çevreleyen kaynak metin.
- **Dynamic context**: İfadeyi değerlendiren invocation'ın durumu ve çalıştırma bağlamı.

Belirli bir ifadenin değerlendirmesinden çıkabilecek değerler her zaman belirli bir WGSL tipine ait olup bu tip ifadenin **static type**'ıdır. Static type yalnızca static context'e bağlıdır.

Bir **type assertion** `e : T`, "*T*, WGSL ifadesi *e*'nin static type'ıdır" demektir. Bu bir runtime kontrolü değil, program metni hakkında bir olgudur.

**Type checking**, başarıyla parse edilmiş bir WGSL modülündeki her ifadeyi static type'ına eşleme ve her statement'ın tip gereksinimlerini doğrulama sürecidir. Başarısız olursa **type error** (bir tür shader-creation error) oluşur.

WGSL **statically typed** bir dildir: type checking yalnızca program kaynak metnini inceleyerek başarılı olur veya type error bulur.

#### 6.1.1 Type Rule Tables

WGSL'in type rule'ları, ifadeler için **type rule table**'lara organize edilir (satır başına bir rule). Bir type rule iki parçadan oluşur:

- **Conclusion** — İfade için type assertion (veya statement için birden fazla type assertion).
- **Preconditions** — Subexpression'lar için type assertion'lar, statement'ın sözdizimsel formu ve diğer koşullar.

**Parameterized** type rule, type parameter içerir. Belirli tipler **substitution** ile atanarak **fully elaborated** rule elde edilir. Her olası substitution bir **overload** üretir.

Örneğin, mantıksal negasyon `!e` type rule'u:

| Precondition | Conclusion |
|-------------|------------|
| *e*: *T*, *T* is `bool` or `vecN<bool>` | `!`*e*: *T* |

Bu parameterized rule'un 4 overload'u vardır (`bool`, `vec2<bool>`, `vec3<bool>`, `vec4<bool>`).

Bir syntactic phrase analiz edilirken üç durum olabilir:
1. **Hiçbir type rule uygulanmaz** → type error.
2. **Tam bir fully elaborated rule uygulanır** → static type belirlenir.
3. **Birden fazla rule uygulanır** → **overload resolution** (§6.1.3) ile çözülür.

#### 6.1.2 Conversion Rank

Bir type assertion `e: T` precondition olarak kullanıldığında şu durumlarda karşılanır:
- *e* zaten *T* tipindeyse, veya
- *e*, *S* tipindeyse ve *S*'den *T*'ye **feasible automatic conversion** varsa.

**ConversionRank(Src, Dest)** fonksiyonu, bir tipten diğerine otomatik dönüşümün tercih edilirliğini ve uygulanabilirliğini ifade eder. Düşük rank daha tercih edilirdir.

| Src → Dest | Rank | Açıklama |
|-----------|------|----------|
| *T* → *T* | 0 | Identity. Dönüşüm yok. |
| `ref<AS,T,AM>` → *T* | 0 | Load Rule — bellekten değer yükleme (AM `read` veya `read_write` ise). |
| AbstractFloat → f32 | 1 | |
| AbstractFloat → f16 | 2 | |
| AbstractInt → i32 | 3 | Değer i32 aralığında değilse shader-creation error. |
| AbstractInt → u32 | 4 | Değer u32 aralığında değilse shader-creation error. |
| AbstractInt → AbstractFloat | 5 | |
| AbstractInt → f32 | 6 | AbstractInt → AbstractFloat → f32 zinciri. |
| AbstractInt → f16 | 7 | AbstractInt → AbstractFloat → f16 zinciri. |
| `vecN<S>` → `vecN<T>` | ConversionRank(S, T) | Component type'tan miras alır. |
| `matCxR<S>` → `matCxR<T>` | ConversionRank(S, T) | Component type'tan miras alır. |
| `array<S,N>` → `array<T,N>` | ConversionRank(S, T) | Component type'tan miras alır (yalnızca fixed-size array). |
| Diğer *S* → *T* | ∞ | Diğer tipler arasında otomatik dönüşüm yoktur. |

**Concretization**: *T* tipi *S*'nin concretization'ıdır eğer *T* concrete ise, reference type değilse, ConversionRank(S,T) sonlu ise ve başka hiçbir concrete non-reference *T2* tipi için ConversionRank(S,T2) ≤ ConversionRank(S,T) değilse.

> **Not:** f32'ye dönüşüm her zaman f16'ya tercih edilir. Otomatik dönüşüm yalnızca modülde `enable f16` aktifse f16 üretir.

#### 6.1.3 Overload Resolution

Birden fazla type rule bir syntactic phrase'e uygulandığında, **overload resolution** ile bağ çözülür:

1. Her overload candidate *C* için, her subexpression `i` konumundaki ConversionRank `C.R(i)` hesaplanır.
2. Subexpression'lardan biri abstract tipe resolve olurken diğeri const-expression değilse, o candidate elimine edilir.
3. *C1*, *C2*'ye **preferred** ise:
   - Her `i` için `C1.R(i) ≤ C2.R(i)`, ve
   - En az bir `i` için `C1.R(i) < C2.R(i)`.
4. Tek bir preferred candidate varsa overload resolution başarılı olur; yoksa başarısız olur → type error.

**Örnekler:**

```wgsl
// log2(32): AbstractInt 32 → AbstractFloat'a dönüştürülür, sonuç AbstractFloat.
let x = log2(32);

// 1 + 2.5: AbstractInt 1 → AbstractFloat 1.0'a dönüşüm (rank 5,0 → kazanır),
// sonuç AbstractFloat.
let y = 1 + 2.5;

// let x = 1 + 2.5: x abstract olamaz, f32 veya f16 kandidatları kalır,
// f32 tercih edilir. Sonuç: let x: f32 = 1.0f + 2.5f;
let z = 1 + 2.5;

// 1u + 2.5: u32'den float'a feasible conversion yok → shader-creation error.
```

---

### 6.2 Plain Types

**Plain type**, boolean, sayı, vektör, matris veya bu değerlerin bileşimlerinin makine temsili için kullanılan tiplerdir.

Plain type = **scalar** | **atomic** | **composite** type.

> **Not:** WGSL'deki plain type'lar C++'daki POD tiplerine benzer, ancak atomic ve abstract numeric tipleri de içerir.

#### 6.2.1 Abstract Numeric Types

Bu tipler WGSL kaynağında yazılamaz; yalnızca type checking tarafından kullanılır. Belirli ifadeler shader-creation time'da, GPU'nun doğrudan implemente ettiğinden daha geniş bir sayısal aralık ve hassasiyetle değerlendirilir.

| Tip | Tanım |
|-----|-------|
| **AbstractInt** | 64-bit two's complement formatında temsil edilebilen tamsayılar. |
| **AbstractFloat** | IEEE-754 binary64 (double precision) formatında temsil edilebilen sonlu floating point sayılar. |

- Abstract numeric tipdeki bir expression değerlendirmesi overflow yapmamalı, sonsuz veya NaN değer üretmemelidir.
- **Abstract** tip: abstract numeric tip içeren veya olan tip. **Concrete** tip: abstract olmayan tip.

Suffix'siz literal'lar abstract numeric tiplere karşılık gelir:
- Suffix'siz integer literal (`123`) → `AbstractInt`
- Suffix'siz float literal (`1.5`) → `AbstractFloat`

```wgsl
// Explicitly-typed
var u32_1 = 1u;     // u32
var i32_1 = 1i;     // i32
var f32_1 = 1f;     // f32

// Inferred — abstract → concrete
let some_i32 = 1;           // → i32 (en tercih edilen: rank 3)
var u32_from_type: u32 = 1; // → u32
var f32_promotion: f32 = 1; // → f32

// Invalid
var i32_demotion: i32 = 1.0; // AbstractFloat → i32 dönüşüm yok!
```

#### 6.2.2 Boolean Type

`bool` tipi `true` ve `false` değerlerini içerir.

| Precondition | Conclusion | Description |
|-------------|------------|-------------|
| — | `true`: bool | True değeri |
| — | `false`: bool | False değeri |

#### 6.2.3 Integer Types

| Tip | Genişlik | Açıklama | Min | Max |
|-----|----------|----------|-----|-----|
| `i32` | 32-bit | Signed integer (two's complement) | `i32(-2147483648)` | `2147483647i` |
| `u32` | 32-bit | Unsigned integer | `0u` | `4294967295u` |

Concrete integer tiplerdeki overflow eden ifadeler, sonucu **modulo 2^bitwidth** üretir.

> **Not:** `AbstractInt` de bir integer tipidir.

#### 6.2.4 Floating Point Types

| Tip | Format | Açıklama |
|-----|--------|----------|
| `f32` | IEEE-754 binary32 (single precision) | Her zaman kullanılabilir. |
| `f16` | IEEE-754 binary16 (half precision) | `enable f16;` directive'i gerektirir, yoksa shader-creation error. |

**Extreme değerler (pozitif):**

| Tip | En küçük subnormal | En küçük normal | En büyük finite | En büyük 2 kuvveti |
|-----|-------------------|----------------|-----------------|-------------------|
| f32 | `0x1p-149f` | `0x1p-126f` | `0x1.fffffep+127f` | `0x1p+127f` |
| f16 | `0x1p-24h` | `0x1p-14h` | `0x1.ffcp+15h` (`65504.0h`) | `0x1p+15h` |

> **Not:** `AbstractFloat` da bir floating point tipidir.

#### 6.2.5 Scalar Types

Scalar tipler: `bool`, `AbstractInt`, `AbstractFloat`, `i32`, `u32`, `f32`, `f16`.

Alt grupları:
- **Numeric scalar**: `AbstractInt`, `AbstractFloat`, `i32`, `u32`, `f32`, `f16`
- **Integer scalar**: `AbstractInt`, `i32`, `u32`

**Scalar conversion**, bir scalar tipteki değeri farklı bir scalar tipe eşler. Şu yollarla gerçekleşir:
- Açık **value constructor** çağrısı.
- Abstract numeric tipten başka bir tipe **feasible automatic conversion** yoluyla.

#### 6.2.6 Vector Types

**Vector**, 2, 3 veya 4 scalar bileşenden oluşan gruplanmış bir dizidir.

```wgsl
vec2<f32>  // 2 bileşenli f32 vektörü
vec3<i32>  // 3 bileşenli i32 vektörü
```

| Tip | Açıklama |
|-----|----------|
| `vecN<T>` | *N* bileşenli *T* tipi vektör. *N* ∈ {2, 3, 4}, *T* scalar olmalıdır. *T*, vektörün **component type**'ıdır. |

Kullanım alanları: yön + büyüklük, uzayda konum, renk temsili (RGBA).

Birçok vektör (ve matris) operasyonu **component-wise** çalışır:

```wgsl
let x: vec3<f32> = a + b; // a, b vec3<f32>
// x[0] = a[0] + b[0]
// x[1] = a[1] + b[1]
// x[2] = a[2] + b[2]
```

**Predeclared vector alias'ları:**

| Alias | Orijinal Tip | Kısıtlama |
|-------|-------------|-----------|
| `vec2i` / `vec3i` / `vec4i` | `vec2<i32>` / `vec3<i32>` / `vec4<i32>` | — |
| `vec2u` / `vec3u` / `vec4u` | `vec2<u32>` / `vec3<u32>` / `vec4<u32>` | — |
| `vec2f` / `vec3f` / `vec4f` | `vec2<f32>` / `vec3<f32>` / `vec4<f32>` | — |
| `vec2h` / `vec3h` / `vec4h` | `vec2<f16>` / `vec3<f16>` / `vec4<f16>` | f16 extension gerekir |

#### 6.2.7 Matrix Types

**Matrix**, 2, 3 veya 4 floating point vektörden oluşan gruplanmış bir dizidir.

| Tip | Açıklama |
|-----|----------|
| `matCxR<T>` | *C* sütun, *R* satırlı matris. *C*, *R* ∈ {2, 3, 4}. *T*, `f32`, `f16` veya `AbstractFloat` olmalıdır. *C* adet `vecR<T>` sütun vektörü olarak görülebilir. |

```wgsl
mat2x3<f32> // 2 sütun, 3 satır f32 matrisi
            // Eşdeğer: 2 adet vec3<f32> sütun vektörü.
```

Temel kullanım: **lineer dönüşüm**. Çarpım operatörü (`*`) ile:
- Scalar ile ölçekleme
- Vektöre dönüşüm uygulama
- Matris birleştirme

**Predeclared matrix alias'ları:**

| f32 Alias'ları | f16 Alias'ları (f16 ext. gerekir) |
|---|---|
| `mat2x2f`, `mat2x3f`, `mat2x4f` | `mat2x2h`, `mat2x3h`, `mat2x4h` |
| `mat3x2f`, `mat3x3f`, `mat3x4f` | `mat3x2h`, `mat3x3h`, `mat3x4h` |
| `mat4x2f`, `mat4x3f`, `mat4x4f` | `mat4x2h`, `mat4x3h`, `mat4x4h` |

#### 6.2.8 Atomic Types

**Atomic type**, concrete integer scalar tipi kapsülleyerek eşzamanlı gözlemcilere belirli garantiler sağlar.

| Tip | Açıklama |
|-----|----------|
| `atomic<T>` | *T* `u32` veya `i32` olmalıdır. |

Kurallar:
- Bir expression atomic tipe evaluate **olmamalıdır**.
- Atomic tipler yalnızca `workgroup` address space'deki değişkenlerde veya `read_write` erişim modlu **storage buffer** değişkenlerinde instantiate edilebilir.
- `workgroup` address space → memory scope `Workgroup`.
- `storage` address space → memory scope `QueueFamily`.
- Geçerli operasyonlar yalnızca **atomic builtin function**'lardır.
- **Atomic modification**: atomic nesnenin içeriğini ayarlayan herhangi bir operasyon. Yeni değer mevcut değerle aynı olsa bile modification sayılır.
- Atomic modification'lar her nesne için **mutually ordered**'dır.

#### 6.2.9 Array Types

**Array**, indeksli bir eleman değerleri dizisidir. İlk eleman index 0'dadır.

| Tip | Açıklama |
|-----|----------|
| `array<E, N>` | **Fixed-size array** — *N* elemanlı, *E* tipli. *N*, **element count**'tur. |
| `array<E>` | **Runtime-sized array** — eleman sayısı çalışma zamanında belirlenir. Yalnızca belirli bağlamlarda kullanılabilir. |

**Fixed-size array *N* kısıtlamaları:**
- *N*, **override-expression** olmalıdır.
- Concrete integer scalar tipe evaluate etmelidir.
- *N* ≤ 0 ise:
  - const-expression ise → shader-creation error
  - Değilse → pipeline-creation error

**Runtime-sized array** eleman sayısı, ilişkili storage buffer binding'in boyutuna göre belirlenir.

**Eleman tipi *E* kısıtlamaları:**
- Plain, constructible veya fixed-footprint tipte olmalıdır.
- Runtime-sized array, struct'ın son üyesinde veya doğrudan storage buffer variable türü olarak kullanılabilir.

```wgsl
array<f32, 4>     // 4 elemanlı f32 dizisi
array<vec3<f32>>  // runtime-sized vec3<f32> dizisi
```

#### 6.2.10 Structure Types

**Structure** (struct), her biri adı ve tipine sahip bir veya daha fazla **member**'dan oluşan, isimlendirilen bir tipdir.

```wgsl
struct PointLight {
  position : vec3f,
  color    : vec3f,
}

struct LightStorage {
  pointCount : u32,
  point      : array<PointLight>,
}
```

Kurallar:
- Her struct `struct` keyword'ü ile bilddirilir.
- Bir WGSL modülünde aynı struct adı en fazla bir kez bildirilebilir.
- İki farklı struct declaration'ı aynı yapıda member'lara sahip olsa bile farklı tiplerdir (**nominal typing**).
- Üyeler bildirim sırasına göre indekslidir (ilk üye = index 0).
- Member erişimi `.member_name` ile yapılır.
- Struct'lar doğrudan veya dolaylı olarak kendi kendine referans veremez (öz-referanslı yapılar **yasaktır**).

#### 6.2.11 Composite Types

**Composite type**, bileşenlerden oluşan tiplerdir:
- Vector — component'ları scalar.
- Matrix — component'ları vector (veya column olarak bakıldığında: scalar'lar).
- Array — element'leri.
- Structure — member'ları.

Her composite tip, bileşenlerine erişim ve doğrudan veya dolaylı olarak scalar tiplerine ulaşmayı sağlar.

**Nesting depth** kavramı: Scalar/atomic = 0; composite tipte `1 + max(bileşenlerin derinliği)`. Maximum nesting depth: **15**.

#### 6.2.12 Constructible Types

**Constructible type**, `let`, `const` veya value constructor ile değer oluşturulabilen tiplerdir:
- `bool`
- Integer scalar tipler: `i32`, `u32`
- Floating point scalar tipler: `f32`, `f16`
- Constructible component tipli vektörler
- Constructible column tipli matrisler
- Constructible eleman tipli fixed-size array'ler
- Tüm üyeleri constructible olan struct'lar

**Constructible olmayan** tipler: `atomic<T>`, runtime-sized array, constructible olmayan üyeye sahip struct.

#### 6.2.13 Fixed-Footprint Types

**Fixed-footprint** tip, boyutu yalnızca tipine göre belirlenen tipdir (runtime bilgisine bağlı değil):

- Scalar tipler, vektörler, matrisler, atomic tipler
- Fixed-size array (element count const-expression ise)
- Tüm üyeleri fixed-footprint olan struct'lar

**Creation-fixed footprint**: element count override-expression veya const-expression ise. **Override-sized** array: element count const-expression olmayan override-expression ise.

---

### 6.3 Enumeration Types

**Enumeration type**, ayrık (discrete) değerler kümesidir. Enum'ın her bir değeri bir **enumerant value**'dur.

Bazı enum tipler **predeclared** olup WGSL kaynak metninde yazılamaz:
- `address_space` — Adres alanı (`function`, `private`, `workgroup`, `uniform`, `storage`, `handle`)
- `access_mode` — Erişim modu (`read`, `write`, `read_write`)
- `texel_format` — Texel biçimi (`rgba8unorm`, `rgba16float`, `r32float` vb.)

#### 6.3.1 Predeclared Enumerants

**Address Space enumerant'ları:**

| Enumerant | Açıklama |
|-----------|----------|
| `function` | Fonksiyon çağrısının ömrü boyunca var olan değişkenler |
| `private` | Invocation'a özel değişkenler |
| `workgroup` | Compute shader workgroup'undaki paylaşılmış değişkenler |
| `uniform` | Uniform buffer değişkenleri (salt okunur) |
| `storage` | Storage buffer değişkenleri |
| `handle` | Texture ve sampler'lar (kaynağa doğrudan yazılamaz) |

**Access Mode enumerant'ları:**

| Enumerant | Açıklama |
|-----------|----------|
| `read` | Yalnızca okuma |
| `write` | Yalnızca yazma |
| `read_write` | Hem okuma hem yazma |

**Texel Format enumerant'ları:**

| Kanal | Formatlar |
|-------|----------|
| RGBA 8-bit | `rgba8unorm`, `rgba8snorm`, `rgba8uint`, `rgba8sint` |
| RGBA 16-bit | `rgba16uint`, `rgba16sint`, `rgba16float` |
| RGBA 32-bit | `rgba32uint`, `rgba32sint`, `rgba32float` |
| RG 32-bit | `rg32uint`, `rg32sint`, `rg32float` |
| R 32-bit | `r32uint`, `r32sint`, `r32float` |
| BGRA 8-bit | `bgra8unorm` |

---

### 6.4 Memory Views

Bellek görünümü tipleri, bellekte saklanan değerlere erişim biçimlerini tanımlar. Bellek görünümü, saklama, paylaşma ve erişim mekanizmalarını kapsar.

#### 6.4.1 Storable Types

**Storable type**, bellekte saklanabilen tiplerdir. Tüm plain tipler (atomic, constructible vb.) ve saklama desteği olan tipler storable'dır.

Storable tip: plain type veya texture/sampler tipi.

#### 6.4.2 Host-shareable Types

**Host-shareable type**, GPU ile CPU arasında paylaşılabilen tiplerdir. `uniform` ve `storage` address space'lerdeki değişkenlerin tipleri host-shareable olmalıdır.

Host-shareable tipler:
- Scalar tipler (abstract olmayan): `i32`, `u32`, `f32`, `f16`
- Atomic tipler
- Host-shareable component tipli vektörler
- Host-shareable column tipli matrisler
- Host-shareable eleman tipli fixed-size array'ler
- Tüm üyeleri host-shareable olan struct'lar

> **Not:** `bool` tipi host-shareable **değildir**.

Bellekteki yerleşim (layout), **alignment** ve **size** kurallarına tabidir. `@align` ve `@size` attribute'ları ile kontrol edilebilir.

#### 6.4.3 Reference and Pointer Types

WGSL'de değişkenler bellekte yer kaplar. Belirli bir bellek konumuna erişmek için iki tip vardır:

| Tip | Açıklama |
|-----|----------|
| `ref<AS, T, AM>` | **Reference type** — kaynak metninde yazılamaz. Bir bellek konumuna doğrudan referans. |
| `ptr<AS, T, AM>` | **Pointer type** — kaynak metninde yazılabilir. Bir bellek konumunun saklanabilir temsili. |

Parametreler:
- *AS*: address space (`function`, `private`, `workgroup`, `uniform`, `storage`)
- *T*: stored type (saklanacak tip)
- *AM*: access mode (`read`, `write`, `read_write`)

Reference tipler otomatik olarak oluşturulur: bir değişken adı kullanıldığında ortaya çıkan ifade bir reference'tır. **Load Rule**, readable reference'tan değer okumak için otomatik olarak uygulanır.

```wgsl
var x: f32 = 1.0;     // x'in tipi: ref<function, f32, read_write>
let value = x;         // Load Rule uygulanır → value: f32
let p = &x;            // p'nin tipi: ptr<function, f32, read_write>
let value2 = *p;       // Pointer dereference → Load Rule → f32
```

#### 6.4.4 Valid and Invalid Memory References

**Valid memory reference**: referansın işaret ettiği bellek konumunun anlamlı değerler barındırması garanti edilen referans.

**Invalid memory reference**: geçersiz referanslara erişim WGSL tarafından tanımsız davranış **değildir** — bunun yerine güvenli varsayılan davranışlar uygulanır.

- Array sınırı dışı erişim → sınır içine normalize edilir (clamp).
- Geçersiz pointer → implementasyon tarafından iyi tanımlanmış bir sonuç üretir.

#### 6.4.5 Originating Variable

Her reference veya pointer'ın bir **originating variable**'ı vardır — bu, referansın nihayetinde işaret ettiği kök değişkendir.

Kurallar:
- Bir fonksiyona pointer parametresi geçirildiğinde, originating variable korunur.
- İki pointer, aynı originating variable'a sahip olmalıdır — aksi durumda birlikte kullanılamazlar.

#### 6.4.6 Out-of-Bounds Access

Dinamik olarak belirlenen bir indeks ile diziye erişildiğinde, indeks sınır dışındaysa davranış:

- **Okuma**: [0, N-1] aralığına clamp edilir (N = eleman sayısı). Sonuç, dizinin geçerli bir elemanının değeridir.
- **Yazma**: [0, N-1] aralığına clamp edilir. Geçerli bir elemana yazma gerçekleşir.
- Bu, hafıza güvenliğini garanti eder ama mantıksal hatayı gizleyebilir.

```wgsl
var arr: array<f32, 4>;
let idx = 10u;
let val = arr[idx]; // idx clamp → arr[3]
```

#### 6.4.7 Use Cases for References and Pointers

**Reference kullanım alanları:**
- Değişkenin değerini okuma (Load Rule otomatik uygulanır).
- Değişkene atama (sol tarafta kullanım).
- Compound assignment (`x += 1`).
- Atomic built-in fonksiyon çağrıları.

**Pointer kullanım alanları:**
- Fonksiyonlara bellek konumu geçirme (pass-by-reference semantiği).
- Referansı saklama ve sonra kullanma.
- Struct veya array üyesine dolaylı erişim.

```wgsl
fn increment(p: ptr<function, i32>) {
  *p += 1;
}

fn main() {
  var counter: i32 = 0;
  increment(&counter);
  // counter artık 1
}
```

#### 6.4.8 Forming Reference and Pointer Values

Reference ve pointer değerleri oluşturma yolları:

| Kaynak | Sonuç Tipi | Açıklama |
|--------|------------|----------|
| Değişken adı `v` | `ref<AS, T, AM>` | Değişkenin referansı |
| `&e` (address-of) | `ptr<AS, T, AM>` | Reference'tan pointer oluşturma |
| `*p` (dereference) | `ref<AS, T, AM>` | Pointer'dan reference oluşturma |
| `e.member` | `ref<AS, M, AM>` | Structure member erişimi |
| `e[i]` | `ref<AS, E, AM>` | Array veya vector element erişimi |

Kısıtlamalar:
- `let` ile bildirilen pointer'lar yalnızca `function` ve `private` address space'lere işaret edebilir.
- Pointer'a pointer oluşturulamaz (çift indirection yok).
- Pointer, global scope'ta `let` ile bildirilemez.

#### 6.4.9 Comparison with References and Pointers in Other Languages

| Özellik | WGSL | C / C++ | Rust |
|---------|------|---------|------|
| Pointer aritmetiği | ❌ Yok | ✅ Var | ❌ Yok (safe) |
| Null pointer | ❌ Yok | ✅ Var | ❌ (safe) |
| Dangling pointer | ❌ Sistem engeller | ⚠ Tanımsız davranış | ❌ Borrow checker engeller |
| Implicit load | ✅ Load Rule | ❌ Explicit dereference | ✅ Auto-deref |
| Tip/erişim kaynak metninde | ✅ address_space + access_mode | ❌ | ✅ Lifetime + mutability |

---

### 6.5 Texture and Sampler Types

Texture ve sampler tipleri, yaygın GPU rendering donanımını WGSL'de kullanıma açar. Her ikisi de **opaque** tiplerdir — yalnızca handle olarak kullanılır, iç yapılarına doğrudan erişilemez.

Texture, GPU belleğindeki çok boyutlu veri dizisidir. Sampler, texture'ın nasıl örnekleneceğini (sample) tanımlar: filtreleme, adres sarmalama vb.

Her ikisi de `handle` address space'te bulunur ve pipeline creation sırasında **binding** ile ilişkilendirilir.

#### 6.5.1 Texel Formats

Texel format, bir texel'in (texture element) bellek düzenini tanımlar. Storage texture'lar belirli bir texel formatı belirtmelidir.

| Format | Bileşenler | Kanal düzeni | Channel format |
|--------|-----------|-------------|---------------|
| `rgba8unorm` | 4 | R8 G8 B8 A8 | [0.0, 1.0] normalized |
| `rgba8snorm` | 4 | R8 G8 B8 A8 | [-1.0, 1.0] normalized |
| `rgba8uint` | 4 | R8 G8 B8 A8 | unsigned integer |
| `rgba8sint` | 4 | R8 G8 B8 A8 | signed integer |
| `rgba16uint` | 4 | R16 G16 B16 A16 | unsigned integer |
| `rgba16sint` | 4 | R16 G16 B16 A16 | signed integer |
| `rgba16float` | 4 | R16 G16 B16 A16 | float |
| `r32uint` | 1 | R32 | unsigned integer |
| `r32sint` | 1 | R32 | signed integer |
| `r32float` | 1 | R32 | float |
| `rg32uint` | 2 | R32 G32 | unsigned integer |
| `rg32sint` | 2 | R32 G32 | signed integer |
| `rg32float` | 2 | R32 G32 | float |
| `rgba32uint` | 4 | R32 G32 B32 A32 | unsigned integer |
| `rgba32sint` | 4 | R32 G32 B32 A32 | signed integer |
| `rgba32float` | 4 | R32 G32 B32 A32 | float |
| `bgra8unorm` | 4 | B8 G8 R8 A8 | [0.0, 1.0] normalized |

Her texel formatına karşılık gelen bir **channel format** ve **texel type** vardır. Texel type, built-in fonksiyonlarda dönüş veya parametre tipi olarak kullanılır.

#### 6.5.2 Sampled Texture Types

**Sampled texture**, shader'da örnekleme veya yükleme (load) ile okunur. Texel component tipi *T*, `f32`, `i32` veya `u32` olmalıdır.

| Tip | Açıklama |
|-----|----------|
| `texture_1d<T>` | 1 boyutlu sampled texture |
| `texture_2d<T>` | 2 boyutlu sampled texture |
| `texture_2d_array<T>` | 2 boyutlu sampled texture dizisi |
| `texture_3d<T>` | 3 boyutlu sampled texture |
| `texture_cube<T>` | Küp haritası sampled texture |
| `texture_cube_array<T>` | Küp haritası sampled texture dizisi |

#### 6.5.3 Multisampled Texture Types

Multisampled texture, her texel'de birden fazla sample barındırır (anti-aliasing için).

| Tip | Açıklama |
|-----|----------|
| `texture_multisampled_2d<T>` | 2 boyutlu multisampled texture, *T*: `f32`, `i32` veya `u32`. |

#### 6.5.4 External Sampled Texture Types

| Tip | Açıklama |
|-----|----------|
| `texture_external` | Dış kaynaklı (video frame vb.) sampled texture. Örnekleme veya yükleme ile okunabilir. |

#### 6.5.5 Storage Texture Types

Storage texture'lar, texel formatı ve erişim modunu açıkça belirtir. Shader tarafından okunabilir veya yazılabilir.

| Tip | Açıklama |
|-----|----------|
| `texture_storage_1d<F, A>` | 1 boyutlu storage texture |
| `texture_storage_2d<F, A>` | 2 boyutlu storage texture |
| `texture_storage_2d_array<F, A>` | 2 boyutlu storage texture dizisi |
| `texture_storage_3d<F, A>` | 3 boyutlu storage texture |

- *F*: texel format (`rgba8unorm`, `r32float` vb.)
- *A*: access mode (`read`, `write`, `read_write`)

#### 6.5.6 Depth Texture Types

Depth texture'lar derinlik bilgisi saklar. Component tipi `f32`'dir.

| Tip | Açıklama |
|-----|----------|
| `texture_depth_2d` | 2 boyutlu depth texture |
| `texture_depth_2d_array` | 2 boyutlu depth texture dizisi |
| `texture_depth_cube` | Küp haritası depth texture |
| `texture_depth_cube_array` | Küp haritası depth texture dizisi |
| `texture_depth_multisampled_2d` | 2 boyutlu multisampled depth texture |

#### 6.5.7 Sampler Type

Sampler, texture örneklemesinde filtreleme ve adres sarmalama gibi parametreleri tanımlar.

| Tip | Açıklama |
|-----|----------|
| `sampler` | Örnekleme sampler'ı (texture filtreleme) |
| `sampler_comparison` | Karşılaştırma sampler'ı (derinlik karşılaştırması için) |

---

### 6.6 AllTypes Type

**AllTypes**, tüm WGSL tiplerinin üst kümesidir. Bu bir kavramsal çerçevedir ve kaynak metninde yazılmaz. Type rule'larda, bir parametrenin herhangi bir tip olabileceğini belirtmek için kullanılır.

---

### 6.7 Type Aliases

**Type alias**, mevcut bir tipe yeni bir isim verir. Alias, orijinal tipin tam eşdeğeridir (nominal olarak farklı **değildir**).

```wgsl
alias float4 = vec4<f32>;
alias Arr = array<i32, 5>;

// Kullanım
var color: float4;
var data: Arr;
```

```bnf
type_alias_decl :
  'alias' ident '=' type_specifier
```

Type alias'lar sadece modül scope'ta bildirilebilir. Bir alias, tanımlandıktan sonra görünür hale gelir. Döngüsel alias zinciri (A → B → A) **yasaktır** ve shader-creation error üretir.

---

### 6.8 Type Specifier Grammar

Tip belirtme (type specifier), bir tipi kaynak metninde ifade etmenin sözdizimsel yoludur.

```bnf
type_specifier :
  | ident template_elaborated_ident
```

Bir type specifier, ya doğrudan bir tip adıdır ya da template list ile parametrelendirilmiş bir formda yazılır:

```wgsl
f32                  // Basit tip adı
vec3<f32>            // Template parametreli tip
array<vec2<f32>, 4>  // İç içe template parametreler
```

---

### 6.9 Predeclared Types and Type-Generators Summary

Tüm predeclared tiplerin ve type-generator'ların özet tablosu:

| Kategori | Tipler / Generator'lar |
|----------|----------------------|
| **Scalar** | `bool`, `i32`, `u32`, `f32`, `f16` |
| **Vector** | `vec2<T>`, `vec3<T>`, `vec4<T>` + alias'lar (`vec2f`, `vec3u` vb.) |
| **Matrix** | `mat2x2<T>` ... `mat4x4<T>` + alias'lar (`mat4x4f`, `mat3x3h` vb.) |
| **Array** | `array<E, N>`, `array<E>` |
| **Atomic** | `atomic<T>` |
| **Pointer** | `ptr<AS, T, AM>` |
| **Texture** | `texture_1d<T>`, `texture_2d<T>`, `texture_3d<T>`, `texture_cube<T>`, `texture_2d_array<T>`, `texture_cube_array<T>`, `texture_multisampled_2d<T>`, `texture_external`, `texture_storage_*<F,A>`, `texture_depth_*` |
| **Sampler** | `sampler`, `sampler_comparison` |

---

> **Önceki:** [← Temeller ve Yapı](01-temeller-ve-yapi.md) · **Sonraki:** [Değişkenler ve İfadeler →](03-degiskenler-ve-ifadeler.md)
