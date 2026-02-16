---
title: Değişkenler ve İfadeler
section: 7-8
source: "W3C WGSL Spec §7–§8"
---

# 3. Değişkenler ve İfadeler

> Verinin nasıl tanımlandığı ve işlendiği: declaration semantiği, address space'ler ve ifade türleri.

---

## §7 Variable and Value Declarations

WGSL'de veri iki temel yolla tanımlanır:
- **Value declarations** (`const`, `override`, `let`): Değeri sabitlenen veya bir kez atanan, **değiştirilemeyen** (immutable) bildirimler.
- **Variable declarations** (`var`): Bellekte yer tutan, değeri **değiştirilebilen** (mutable) bildirimler.

Her bildirim bir **ad** (identifier), isteğe bağlı bir **tip belirtici** (type specifier) ve isteğe bağlı bir **başlatıcı** (initializer) ifadesinden oluşur.

### 7.1 Variables vs Values

| Özellik | Value (`const`, `override`, `let`) | Variable (`var`) |
|---------|-----------------------------------|------------------|
| Bellek adresi | Yok (sadece değer) | Var (address space'e bağlı) |
| Mutability | Immutable | Mutable |
| Referans üretilebilir mi? | Hayır | Evet (`ref<AS,T,AM>`) |
| Pointer alınabilir mi? | Hayır | Evet (`&variable`) |

**Temel kural:** Bir *value declaration*'ın adı kullanıldığında doğrudan **değer** elde edilir. Bir *variable*'ın adı kullanıldığında ise o değişkene ait bir **referans** (memory view) elde edilir; değer elde etmek için *load rule* uygulanır.

### 7.2 Value Declarations

#### 7.2.1 `const` Declarations

`const` bildirimi, **shader-creation zamanında** (derleme zamanı) sabitlenen bir değer tanımlar.

**Kurallar:**
- Her zaman bir **başlatıcı** gerektirir.
- Başlatıcı bir **const-expression** olmalıdır.
- **Module scope** veya **function scope**'ta bildirilebilir.
- Effective-value-type **concrete** veya **abstract** olabilir.
  - Tip belirtilmezse, başlatıcının tipi aynen korunur (abstract kalabilir).
  - Tip belirtilirse, başlatıcı o tipe dönüştürülür.

```wgsl
const PI: f32 = 3.14159265;       // Concrete tip: f32
const WORKGROUP_SIZE = 256u;       // Concrete tip: u32 (suffix sayesinde)
const HALF = 0.5;                  // Abstract tip: AbstractFloat
const COUNT = 42;                  // Abstract tip: AbstractInt
```

**Abstract tiplerin avantajı:** `const` bildirimleri abstract kalabildiğinden, kullanıldıkları bağlamda en uygun somut tipe otomatik dönüştürülürler. Bu, daha esnek ve taşınabilir kod yazmaya olanak tanır.

#### 7.2.2 `override` Declarations

`override` bildirimi, **pipeline-creation zamanında** sabitlenen ve API aracılığıyla değiştirilebilen sabitler tanımlar.

**Kurallar:**
- Yalnızca **module scope**'ta bildirilebilir.
- Başlatıcı **isteğe bağlıdır** (API ile değer sağlanabilir).
- Effective-value-type **concrete scalar** olmalıdır (`i32`, `u32`, `f32`, `f16`, `bool`).
- `@id(n)` attribute'u ile API'den erişim için benzersiz bir kimlik atanabilir.

```wgsl
@id(0) override block_size: u32 = 64;       // Pipeline'da override edilebilir
@id(1) override use_hdr: bool = false;       // Boolean override
override intensity: f32 = 1.0;               // @id olmadan da kullanılabilir
override scale = 2.0;                        // Tip f32'ye dönüştürülür
```

**API etkileşimi:** Pipeline oluşturulurken `@id` ile belirtilen sabitlerin değerleri API aracılığıyla değiştirilebilir. Eğer API bir değer sağlamaz ve başlatıcı da yoksa, **pipeline-creation error** oluşur.

#### 7.2.3 `let` Declarations

`let` bildirimi, **çalışma zamanında** (runtime) değeri belirlenen ancak bir kez atandıktan sonra **değiştirilemeyen** değerler tanımlar.

**Kurallar:**
- Yalnızca **function scope**'ta bildirilebilir.
- Başlatıcı **zorunludur**.
- Effective-value-type **concrete constructible** veya **pointer** tipinde olmalıdır.
  - Abstract tipler otomatik olarak concrete'e dönüştürülür (overload resolution ile).
- Her kontrol akışı bildirimi geçtiğinde başlatıcı yeniden değerlendirilir.

```wgsl
let idx = global_invocation_id.x;             // u32 (runtime değer)
let doubled = idx * 2u;                       // Runtime hesaplama
let p: ptr<function, f32> = &some_var;        // Pointer tipi de kabul edilir
let minint = -2147483648;                     // AbstractInt → i32'ye dönüştürülür
```

**`const` vs `let` farkı:**
- `const`: Derleme zamanında değer belirlenir, abstract kalabilir, her yerde kullanılabilir.
- `let`: Çalışma zamanında değer belirlenir, her zaman concrete'tir, sadece fonksiyon içinde geçerlidir.

### 7.3 `var` Declarations

`var` bildirimi, bellekte yer tutan ve değeri **değiştirilebilen** (mutable) bir değişken tanımlar.

**Sözdizimi:**
```
var<address_space, access_mode> name: type = initializer;
```

**Kurallar:**
- Store type bir **concrete constructible** veya **fixed-footprint** tip olmalıdır (adres uzayına bağlı).
- Module scope'ta: `private`, `workgroup`, `uniform`, `storage`, `handle` adres uzayları kullanılabilir.
- Function scope'ta: Varsayılan adres uzayı `function`'dır.

```wgsl
// Function scope (varsayılan: function address space)
var count: u32;                              // Başlatıcı yok → zero value
var delta: i32;                              // Zero value: 0
var sum: f32 = 0.0;                          // Başlatıcı ile
var pi = 3.14159;                            // Tip çıkarımı: f32

// Module scope
var<private> counter: u32 = 0u;              // Private address space
var<workgroup> shared_data: array<f32, 256>; // Workgroup address space

// Resource bindings (module scope, API ile bağlanır)
@group(0) @binding(0)
var<uniform> params: Params;                 // Uniform buffer (read-only)

@group(0) @binding(1)
var<storage, read_write> data: array<f32>;   // Storage buffer (read/write)

@group(0) @binding(2)
var<storage, read> input: array<vec2<f32>>;  // Storage buffer (read-only)

@group(0) @binding(3)
var my_texture: texture_2d<f32>;             // Handle address space (texture)

@group(0) @binding(4)
var my_sampler: sampler;                     // Handle address space (sampler)
```

#### Adres Uzayları ve Erişim Modları

| Adres Uzayı | Scope | Başlatıcı | Varsayılan Erişim | Açıklama |
|-------------|-------|-----------|-------------------|----------|
| `function` | Function | İsteğe bağlı | `read_write` | Fonksiyon-yerel bellek |
| `private` | Module | İsteğe bağlı | `read_write` | İnvocation-yerel bellek |
| `workgroup` | Module | Yok | `read_write` | Workgroup paylaşımlı bellek |
| `uniform` | Module | Yok | `read` | Uniform buffer (salt okunur) |
| `storage` | Module | Yok | `read` | Storage buffer |
| `handle` | Module | Yok | `read` | Texture ve sampler |

#### Başlangıç Değerleri (Default Initial Values)

Başlatıcı belirtilmezse değişkenler **zero value** ile başlatılır:

- **`function` adres uzayı:** Başlatıcı yoksa zero value; varsa çalışma zamanında değerlendirilir.
- **`private` adres uzayı:** Başlatıcı yoksa zero value; varsa override-expression olmalı, pipeline-creation zamanında değerlendirilir.
- **`workgroup` adres uzayı:** Her zaman zero value. Başlatıcı kullanılamaz. Constructible olmayan tipler (ör. `atomic`) için özyinelemeli kurallar uygulanır.
- **Diğer adres uzayları:** (`uniform`, `storage`, `handle`) kaynak bağlamaları (bindings) ile belirlenir.

```wgsl
var i: i32;                 // Başlangıç değeri: 0
loop {
  var twice: i32 = 2 * i;   // Her iterasyonda yeniden değerlendirilir
  i++;
  if i == 5 { break; }
}
// i: 0,1,2,3,4,5 — twice: 0,2,4,6,8
```

### 7.4 Variable and Value Declaration Grammar Summary

```
variable_or_value_statement :
  | variable_decl
  | variable_decl '=' expression
  | 'let' optionally_typed_ident '=' expression
  | 'const' optionally_typed_ident '=' expression

variable_decl :
  | 'var' _disambiguate_template template_list? optionally_typed_ident

optionally_typed_ident :
  | ident ( ':' type_specifier )?

global_variable_decl :
  | attribute* variable_decl ( '=' expression )?

global_value_decl :
  | 'const' optionally_typed_ident '=' expression
  | attribute* 'override' optionally_typed_ident ( '=' expression )?
```

---

## §8 Expressions

İfadeler (expressions), değerlerin nasıl hesaplandığını belirler. WGSL'de ifadeler **üç zamanlama kategorisine** ayrılır:

| Kategori | Değerlendirme Zamanı | Kullanım Alanı |
|----------|---------------------|----------------|
| **const-expression** | Shader-creation zamanı | En geniş: dizi boyutları, const başlatıcılar |
| **override-expression** | Pipeline-creation zamanı | Override başlatıcılar, bazı dizi boyutları |
| **runtime-expression** | GPU shader yürütmesi | En esnek: tüm runtime değerler |

**Temel prensip:** Değerlendirme ne kadar erken yapılırsa, ifade o kadar kısıtlı ama o kadar çok yerde kullanılabilir.

### 8.1 Early Evaluation Expressions

#### 8.1.1 `const` Expressions

Shader-creation zamanında değerlendirilebilen ifadeler **const-expression** olarak adlandırılır. Bir ifadenin const-expression olması için tüm tanımlayıcılarının şunlara çözümlenmesi gerekir:

- `const` bildirimleri
- const-fonksiyonlar (built-in fonksiyonların const-evaluable alt kümesi)
- Tip takma adları (type aliases)
- Yapı (struct) adları

**Değerlendirme kuralları:**
- Bir const-expression *E* yalnızca şu koşullarda değerlendirilir:
  1. *E* bir üst-düzey ifade (top-level expression) ise
  2. *E* başka bir ifadenin alt ifadesi ise ve o üst ifadenin değerlendirilmesi *E*'yi gerektiriyorsa
  3. *E*'nin değerlendirilmesi bir üst ifadenin statik tipini belirlemek için gerekli ise
  4. *E*'nin değerlendirilmesi bir shader-creation error üretmek için gerekli ise

**Kısa devre (short-circuiting):** `&&` ve `||` operatörleri, sağ tarafın tip belirleme gerektirmediği durumlarda sağ tarafı değerlendirmeyebilir.

```wgsl
// Örnek: -2147483648 analizi
const minint = -2147483648;
// 1. "2147483648" → AbstractInt değeri 2147483648
// 2. "-" uygulanır → AbstractInt değeri -2147483648
// 3. const bildirimi abstract tipe izin verir → minint: AbstractInt

// Örnek: let ile fark
let minint2 = -2147483648;
// 1. Aynı AbstractInt değeri -2147483648
// 2. let concrete tip gerektirir → overload resolution
// 3. En düşük rank: i32 → minint2: i32 değeri -2147483648

// Örnek: Kısa devre davranışı
// false && (10i < i32(5 * 1000 * 1000 * 1000))
// Sol taraf false → sağ taraf değerlendirilmez
// i32(5000000000) değerlendirilseydi taşma hatası oluşurdu
```

**Hassasiyet:** Const-expression'lar CPU tarafından değerlendirilebildiğinden, `AbstractFloat` işlemleri için hassasiyet gereksinimleri WebAssembly ve ECMAScript ortamlarından daha katı değildir. Concrete float tipleri (`f32`) için ise §15.7.4.1'deki hassasiyet kuralları geçerlidir.

#### 8.1.2 `override` Expressions

Pipeline-creation zamanında değerlendirilebilen ifadeler **override-expression** olarak adlandırılır. Bir override-expression, const-expression koşullarına ek olarak `override` bildirimlerine de referans verebilir.

> **Not:** Tüm const-expression'lar aynı zamanda override-expression'dır, ancak tersi geçerli değildir.

**Override-expression'lar yalnızca pipeline-creation zamanında doğrulanır ve değerlendirilir**, API tarafından sağlanan değer atamaları sonrasında.

```wgsl
override a: i32 = 0;
override b = 1 / a;   // Pipeline-creation error (a=0 ise)

// b, frag1 shader'ının parçası. Pipeline oluşturulurken:
// - b override edilirse → hata yok
// - a sıfırdan farklı override edilirse → hata yok
// - a=0 ve b override edilmezse → pipeline-creation error
@fragment
fn frag1() {
  _ = b;
}

// b, frag2'nin parçası değil → hiçbir zaman hata oluşmaz
@fragment
fn frag2() { }
```

```wgsl
// override ile tip dönüşümü
override x = 42;               // 42 → AbstractInt → i32 (override concrete gerektirir)
let y = x + 1;                 // override-expression, pipeline-creation zamanında değerlendirilir
let v = vec3(x, x, x);         // Tip: vec3<i32>
```

### 8.2 Indeterminate Values

Sınırlı durumlarda, bir runtime-expression'ın alt ifadesi için desteklenmeyen değerlerle değerlendirilmesi gerekebilir. Bu durumda sonuç, ifadenin statik tipine ait **belirsiz bir değer** (indeterminate value) olur.

**Özellikler:**
- Her benzersiz **dynamic context** (ör. döngü iterasyonu) için farklı bir belirsiz değer üretilebilir.
- Floating-point tipler için belirsiz değer **NaN** olabilir (uygulama destekliyorsa).
- Belirsiz bir değer kendisiyle karşılaştırıldığında bile sonuç tahmin edilemez (NaN ≠ NaN).

```wgsl
fn fun() {
  const v = vec2<i32>(0, 1);
  for (var i: i32 = 0; i < 2; i++) {
    let extract = v[i + 5];      // Sınır dışı → indeterminate value
    // extract herhangi bir i32 değeri olabilir

    if extract == extract {
      // Bu her zaman çalıştırılır (i32 için NaN yok)
    }
  }
}

fn float_fun(runtime_index: u32) {
  const v = vec2<f32>(0, 1);
  let float_extract: f32 = v[runtime_index + 5]; // Indeterminate, NaN olabilir

  if float_extract == float_extract {
    // NaN == NaN daima false olduğundan bu çalıştırılMAYABİLİR
  }
}
```

### 8.3 Literal Value Expressions

Literal değerlerin tip kuralları:

| Literal | Tip | Açıklama |
|---------|-----|----------|
| `true` | `bool` | Boolean true değeri |
| `false` | `bool` | Boolean false değeri |
| `42` (suffix yok) | `AbstractInt` | Soyut tamsayı |
| `3.14` (suffix yok) | `AbstractFloat` | Soyut kayan nokta |
| `42i` | `i32` | 32-bit işaretli tamsayı |
| `42u` | `u32` | 32-bit işaretsiz tamsayı |
| `3.14f` | `f32` | 32-bit kayan nokta |
| `3.14h` | `f16` | 16-bit kayan nokta |

### 8.4 Parenthesized Expressions

Parantez ifadesi `(e)`, `e` ifadesinin tipini ve değerini aynen korur. Operatör önceliğini kontrol etmek veya ifadeyi çevreleyen bağlamdan izole etmek için kullanılır.

```wgsl
let a = (2 + 3) * 4;   // 20 (parantez önceliği belirler)
let b = 2 + 3 * 4;     // 14
```

### 8.5 Composite Value Decomposition Expressions

Bileşik (composite) değerlerin bileşenlerine erişmek için iki yol vardır:

1. **Named component expression** (`.` operatörü): Vektör ve yapı (struct) bileşenlerine isimle erişim.
2. **Indexing expression** (`[]` operatörü): Vektör, matris ve dizi elemanlarına indeksle erişim.

**Sınır kontrolü (bounds checking):** İndeks değeri `i` aralık dışındaysa:
- const-expression ise → **shader-creation error**
- override-expression ise → **pipeline-creation error**
- Runtime ise → **indeterminate value** döner veya **invalid memory reference** oluşur.

#### 8.5.1 Vector Access Expression

Vektör bileşenlerine iki yolla erişilebilir:

**a) İndeksleme:**
```wgsl
var v: vec3<f32> = vec3<f32>(1.0, 2.0, 3.0);
let e: f32 = v[1];           // e = 2.0
```

**b) Swizzle (convenience letters):**

İki set convenience name tanımlıdır:
- **Dimensional set:** `x`, `y`, `z`, `w` → bileşen 0, 1, 2, 3
- **Color set:** `r`, `g`, `b`, `a` → bileşen 0, 1, 2, 3

```wgsl
var a: vec3<f32> = vec3<f32>(1., 2., 3.);
var b: f32 = a.y;               // b = 2.0 (tek bileşen seçimi)
var c: vec2<f32> = a.bb;        // c = (3.0, 3.0) (swizzle, tekrar olabilir)
var d: vec3<f32> = a.zyx;       // d = (3.0, 2.0, 1.0) (sıralama değişebilir)
var e: f32 = a[1];              // e = 2.0 (indeksleme)
```

**Kurallar:**
- İki set **karıştırılamaz**: `.rybw` **geçersizdir**.
- Harf sayısı 1–4 arası olmalıdır.
- Harfler vektörün boyutunu aşmamalıdır (`z` yalnızca `vec3`/`vec4`, `w` yalnızca `vec4`).
- **Tek harf** swizzle skaler sonuç üretir, **çoklu harf** swizzle vektör üretir.

##### 8.5.1.1 Vector Single Component Selection

| İfade | Sonuç Tipi | Açıklama |
|-------|-----------|----------|
| `e.x` / `e.r` | `T` | İlk bileşen |
| `e.y` / `e.g` | `T` | İkinci bileşen |
| `e.z` / `e.b` | `T` | Üçüncü bileşen (N≥3) |
| `e.w` / `e.a` | `T` | Dördüncü bileşen (N=4) |
| `e[i]` | `T` | i'inci bileşen (i: i32/u32) |

##### 8.5.1.2 Vector Multiple Component Selection

Çoklu harf swizzle, kaynak vektörün bileşenlerinden yeni bir vektör oluşturur:

```wgsl
// e: vecN<T> için
e.xy   → vec2<T>     // İlk iki bileşen
e.rrr  → vec3<T>     // İlk bileşeni 3 kez tekrarla
e.wzyx → vec4<T>     // Ters sıra (N=4 gerektirir)
```

> **Kısıtlama:** Çoklu swizzle ifadesi atama (assignment) sol tarafında kullanılamaz çünkü her zaman yeni bir **değer** üretir, referans değil.

##### 8.5.1.3 Component Reference from Vector Memory View

Bir vektör değişkeninin memory view'ından tek bir bileşene referans elde edilebilir:

```wgsl
var v: vec3<f32> = vec3<f32>(1.0, 2.0, 3.0);
v.y = 5.0;               // v'nin ikinci bileşenine yazma (referans üzerinden)
v[0] = 10.0;              // İndeksle bileşen referansı
```

**Önemli:** Bir vektör bileşenine yazma, o vektörün **tüm** bellek konumlarına erişebilir. Farklı invocation'lar aynı vektörün farklı bileşenlerine yazıyorsa senkronizasyon gereklidir.

#### 8.5.2 Matrix Access Expression

Matris erişimi, **sütun vektörü** indekslemesi ile yapılır:

```wgsl
var m: mat2x3<f32> = mat2x3<f32>(1., 2., 3., 4., 5., 6.);
let col: vec3<f32> = m[0];       // İlk sütun vektörü: (1, 2, 3)
let elem: f32 = m[1][2];         // İkinci sütun, üçüncü satır: 6.0
```

**İndeks sınır kontrolü:** `i` değeri [0, C-1] aralığı dışındaysa aynı hata kuralları geçerlidir (shader-creation error, pipeline-creation error veya indeterminate value).

#### 8.5.3 Array Access Expression

```wgsl
var arr: array<f32, 3> = array<f32, 3>(10., 20., 30.);
let val: f32 = arr[1];           // 20.0

// Runtime-sized array (storage buffer)
@group(0) @binding(0)
var<storage> data: array<f32>;
let x = data[global_id.x];       // Runtime indeksleme
```

- **Sabit-boyutlu diziler:** İndeks sınır dışıysa → hata veya indeterminate value.
- **Runtime-sized diziler:** Sınır dışı indeks → invalid memory reference.
  - Negatif işaretli indeks const/override-expression ise derleme/pipeline hatası.

#### 8.5.4 Structure Access Expression

Yapı üyelerine `.` operatörü ile erişilir:

```wgsl
struct Light {
  position: vec3<f32>,
  intensity: f32,
}

var light: Light;
let pos = light.position;       // vec3<f32> değeri
light.intensity = 2.0;          // Üyeye yazma (referans üzerinden)
```

### 8.6 Logical Expressions

#### Tekli (Unary) Mantıksal İşlemler

| İfade | Ön Koşul | Sonuç | Açıklama |
|-------|----------|-------|----------|
| `!e` | `e: T` (T=`bool` veya `vecN<bool>`) | `T` | Mantıksal olumsuzlama. Vektörse bileşen-bazlı. |

#### İkili (Binary) Mantıksal İşlemler

| İfade | Ön Koşul | Sonuç | Açıklama |
|-------|----------|-------|----------|
| `e1 \|\| e2` | `e1: bool`, `e2: bool` | `bool` | **Kısa devre OR.** `e1` true ise `e2` değerlendirilmez. |
| `e1 && e2` | `e1: bool`, `e2: bool` | `bool` | **Kısa devre AND.** `e1` false ise `e2` değerlendirilmez. |
| `e1 \| e2` | `e1: T`, `e2: T` (T=`bool`/`vecN<bool>`) | `T` | Mantıksal OR. Her iki taraf da değerlendirilir. Vektörse bileşen-bazlı. |
| `e1 & e2` | `e1: T`, `e2: T` (T=`bool`/`vecN<bool>`) | `T` | Mantıksal AND. Her iki taraf da değerlendirilir. Vektörse bileşen-bazlı. |

**Kısa devre vs Normal:** `||` ve `&&` yalnızca skaler `bool` üzerinde çalışır ve kısa devre yapar. `|` ve `&` vektör booleanlar üzerinde de çalışır ama kısa devre **yapmaz** (her iki taraf da değerlendirilir).

### 8.7 Arithmetic Expressions

#### Tekli Aritmetik

| İfade | Ön Koşul | Sonuç | Açıklama |
|-------|----------|-------|----------|
| `-e` | `e: T` (T: sayısal skaler veya vektör) | `T` | Negatif alma. İşaretli tamsayıda en küçük negatif değerin negatifi kendisidir. |

> T olabilecek tipler: `AbstractInt`, `AbstractFloat`, `i32`, `f32`, `f16` ve bunların vektör biçimleri. Not: `u32` negatif alınamaz.

#### İkili Aritmetik

S: `AbstractInt`, `AbstractFloat`, `i32`, `u32`, `f32`, `f16`
T: `S` veya `vecN<S>`

| İfade | Sonuç | Açıklama |
|-------|-------|----------|
| `e1 + e2` | `T` | **Toplama.** Vektörse bileşen-bazlı. |
| `e1 - e2` | `T` | **Çıkarma.** Vektörse bileşen-bazlı. |
| `e1 * e2` | `T` | **Çarpma.** Vektörse bileşen-bazlı. |
| `e1 / e2` | `T` | **Bölme.** Vektörse bileşen-bazlı. |
| `e1 % e2` | `T` | **Kalan.** Vektörse bileşen-bazlı. |

**Tamsayı Bölme Kuralları (İşaretli):**
- `e2 = 0` → hata (const/override/runtime bağlamına göre farklı davranış)
- `e1` en küçük negatif ve `e2 = -1` → hata (taşma)
- Diğer durumlar → `truncate(e1 ÷ e2)` (sıfıra doğru yuvarlama)

**Tamsayı Bölme Kuralları (İşaretsiz):**
- `e2 = 0` → hata; runtime'da `e1` döner
- Normal → `e1 = q × e2 + r` ve `0 ≤ r < e2`

**Tamsayı Kalan Kuralları:**
- Sıfıra bölme ve taşma kuralları bölme ile aynıdır.
- Sıfır olmayan sonuç, `e1` ile aynı işareti taşır.

#### Karışık Skaler-Vektör Aritmetiği

Bir skaler ile bir vektör arasında aritmetik yapıldığında, skaler otomatik olarak vektöre broadcast edilir:

```wgsl
let v = vec3<f32>(1.0, 2.0, 3.0);
let s: f32 = 2.0;

let r1 = v * s;    // vec3(2.0, 4.0, 6.0) — v * vec3(s)
let r2 = s + v;    // vec3(3.0, 4.0, 5.0) — vec3(s) + v
```

#### Matris Aritmetiği

T: `AbstractFloat`, `f32`, `f16`

| İfade | Sonuç | Açıklama |
|-------|-------|----------|
| `m1 + m2` | `matCxR<T>` | Bileşen-bazlı toplama |
| `m1 - m2` | `matCxR<T>` | Bileşen-bazlı çıkarma |
| `m * s` / `s * m` | `matCxR<T>` | Skaler-matris çarpma (bileşen-bazlı) |
| `m * v` | `vecR<T>` | Matris × sütun vektörü çarpımı (m: `matCxR`, v: `vecC`) |
| `v * m` | `vecC<T>` | Satır vektörü × matris çarpımı (v: `vecR`, m: `matCxR`) |
| `m1 * m2` | `matCxR<T>` | Lineer cebir matris çarpımı (m1: `matKxR`, m2: `matCxK`) |

### 8.8 Comparison Expressions

S: `AbstractInt`, `AbstractFloat`, `bool`, `i32`, `u32`, `f32`, `f16`
T: `S` veya `vecN<S>`
TB: `vecN<bool>` (T vektörse), `bool` (T skalerse)

| İfade | Sonuç | Açıklama |
|-------|-------|----------|
| `e1 == e2` | `TB` | Eşitlik. Vektörse bileşen-bazlı. |
| `e1 != e2` | `TB` | Eşitsizlik. Vektörse bileşen-bazlı. |
| `e1 < e2` | `TB` | Küçüktür. Vektörse bileşen-bazlı. (`bool` hariç) |
| `e1 <= e2` | `TB` | Küçük eşit. Vektörse bileşen-bazlı. (`bool` hariç) |
| `e1 > e2` | `TB` | Büyüktür. Vektörse bileşen-bazlı. (`bool` hariç) |
| `e1 >= e2` | `TB` | Büyük eşit. Vektörse bileşen-bazlı. (`bool` hariç) |

> **Not:** `<`, `<=`, `>`, `>=` operatörleri `bool` tipi üzerinde **kullanılamaz**; yalnızca `==` ve `!=` kullanılabilir.

**Vektör karşılaştırma sonucu:** Vektörlerde karşılaştırma bileşen-bazlı yapılır ve sonuç `vecN<bool>` tipindedir (skaler bool değil).

### 8.9 Bit Expressions

#### Tekli Bitwise

| İfade | Ön Koşul | Sonuç | Açıklama |
|-------|----------|-------|----------|
| `~e` | `e: T` (S: `AbstractInt`, `i32`, `u32`; T: S veya `vecN<S>`) | `T` | Bitwise tümleyen. Her bit tersine çevrilir. |

#### İkili Bitwise

S: `AbstractInt`, `i32`, `u32`; T: `S` veya `vecN<S>`

| İfade | Sonuç | Açıklama |
|-------|-------|----------|
| `e1 \| e2` | `T` | Bitwise OR |
| `e1 & e2` | `T` | Bitwise AND |
| `e1 ^ e2` | `T` | Bitwise XOR (exclusive or) |

> **Not:** `|` ve `&` operatörleri boolean tipler (mantıksal) ve tamsayı tipler (bitwise) üzerinde aynı sözdizimini paylaşır. Operandın tipi hangisinin uygulanacağını belirler.

#### Bit Kaydırma (Shift)

| İfade | Ön Koşul | Sonuç | Açıklama |
|-------|----------|-------|----------|
| `e1 << e2` | `e1: T`, `e2: TS`<br>(S: `i32`/`u32`, TS: `u32`) | `T` | Sola kaydırma. En az anlamlı bitlere 0 eklenir. |
| `e1 >> e2` | `e1: T`, `e2: TS` | `T` | Sağa kaydırma. İşaretsiz: 0 eklenir. İşaretli: işaret biti korunur (aritmetik kaydırma). |

**Kaydırma kuralları:**
- Kaydırma miktarı `e2` modulo `e1`'in bit genişliğidir.
- `e2 ≥ bit genişliği` ise → const/override bağlamında hata.
- Sola kaydırmada, const/override bağlamında **taşma kontrolü** yapılır:
  - İşaretli: Kaydırılan ve sonucun aynı işaret bitine sahip olması gerekir.
  - İşaretsiz: Kaydırılan bitlerden hiçbiri 1 olmamalıdır.

### 8.10 Function Call Expression

Fonksiyon çağrısı ifadesi, dönüş tipi olan bir fonksiyonu çalıştırır ve sonucu bir değer olarak üretir.

```wgsl
let d = dot(v1, v2);              // Built-in fonksiyon çağrısı
let r = my_function(a, b, c);     // Kullanıcı tanımlı fonksiyon
let v = vec3<f32>(1.0, 2.0, 3.0); // Değer yapıcısı (value constructor)
```

Fonksiyon dönüş değeri yoksa, **function call statement** (§9.5) kullanılmalıdır.

### 8.11 Variable Identifier Expression

Bir `var` değişkeninin adı bir ifadede kullanıldığında, sonuç o değişkenin belleğine ait bir **referans** (reference) olur:

```
v: ref<AS, T, AM>
```

burada `AS` adres uzayı, `T` store type, `AM` erişim modudur.

```wgsl
var x: f32 = 1.0;
// "x" ifadesi → ref<function, f32, read_write>
let y = x;   // Load rule uygulanır → f32 değeri 1.0
x = 2.0;     // Referans üzerinden yazma
```

### 8.12 Formal Parameter Expression

Fonksiyon parametresi olan bir tanımlayıcı, çağrı noktasında (call site) sağlanan değeri üretir.

```wgsl
fn add(a: f32, b: f32) -> f32 {
  return a + b;     // a ve b → formal parameter expression
}
```

### 8.13 Address-Of Expression

`&` (address-of) operatörü, bir referansı (reference) karşılık gelen **pointer** değerine dönüştürür:

```wgsl
var x: f32 = 1.0;
let p: ptr<function, f32> = &x;   // ref → ptr dönüşümü
// p: ptr<function, f32, read_write>
```

**Kısıtlamalar:**
- `handle` adres uzayındaki değişkenler için pointer alınamaz → **shader-creation error**
- Vektör bileşeni referansından pointer alınamaz → **shader-creation error**

### 8.14 Indirection Expression

`*` (indirection) operatörü, bir pointer'ı karşılık gelen **referansa** dönüştürür:

```wgsl
var x: f32 = 1.0;
let p: ptr<function, f32> = &x;
let val = *p;          // *p → ref<function, f32, read_write>, load rule → 1.0
*p = 3.0;              // Pointer üzerinden yazma
```

Invalid memory reference olan bir pointer'ın indirection'ı da invalid memory reference üretir.

### 8.15 Identifier Expressions for Value Declarations

`const`, `override` ve `let` ile tanımlanmış tanımlayıcılar kullanıldığında doğrudan değer üretirler:

| Bildirim | Değerlendirme Zamanı | Sonuç |
|----------|---------------------|-------|
| `const c: T` | Shader-creation | Başlatıcı değeri (const-expression) |
| `override c: T` | Pipeline-creation | API değeri veya başlatıcı |
| `let c: T` | Runtime (kontrol akışı bildirimi geçtiğinde) | Başlatıcı değeri |

### 8.16 Enumeration Expressions

Önceden tanımlanmış (predeclared) numaralandırma değerlerine tanımlayıcıları ile erişilir:

```wgsl
// Bağımsız kullanım (fully qualified)
let fmt = rgba8unorm;         // texel_format numaralandırması

// Template parametresi olarak kullanım
var<storage, read_write> buf: array<f32>;
// "read_write" → access_mode numaralandırma değeri
```

### 8.17 Type Expressions

Tip ifadeleri, WGSL'de bir tipi temsil eden ifadelerdir:

```wgsl
// Predeclared type
let x: f32 = 1.0;

// Type alias
alias Color = vec4<f32>;
let c: Color = Color(1.0, 0.0, 0.0, 1.0);

// Structure type
struct Vertex { pos: vec3<f32> }
let v = Vertex(vec3(0.0));

// Type-generator (template parameters ile)
let v: vec2<f32> = vec2<f32>(1.0, 2.0);
let m: mat4x4<f32> = mat4x4<f32>(1.0, ...);
```

### 8.18 Expression Grammar Summary

WGSL ifade grameri, operatör önceliği ve ilişkilendirme (associativity) kurallarını kodlar:

```
primary_expression :
  | template_elaborated_ident
  | call_expression
  | literal
  | paren_expression

call_expression : call_phrase
call_phrase : template_elaborated_ident argument_expression_list

paren_expression : '(' expression ')'

singular_expression : primary_expression component_or_swizzle_specifier?

unary_expression :
  | singular_expression
  | '-' unary_expression
  | '!' unary_expression
  | '~' unary_expression
  | '*' unary_expression
  | '&' unary_expression

multiplicative_expression :
  | unary_expression
  | multiplicative_expression ('*'|'/'|'%') unary_expression

additive_expression :
  | multiplicative_expression
  | additive_expression ('+'|'-') multiplicative_expression

shift_expression :
  | additive_expression
  | unary_expression '<<' unary_expression
  | unary_expression '>>' unary_expression

relational_expression :
  | shift_expression
  | shift_expression ('<'|'>'|'<='|'>='|'=='|'!=') shift_expression

expression :
  | relational_expression
  | short_circuit_or_expression '||' relational_expression
  | short_circuit_and_expression '&&' relational_expression
  | bitwise_expression
```

### 8.19 Operator Precedence and Associativity

Operatör önceliği tablosu (güçlüden zayıfa):

| Öncelik | İsim | Operatörler | Yön | Bağlama |
|---------|------|-------------|-----|---------|
| 1 | Parenthesized | `(...)` | — | — |
| 2 | Primary | `a()`, `a[]`, `a.b` | Soldan sağa | — |
| 3 | Unary | `-a`, `!a`, `~a`, `*a`, `&a` | Sağdan sola | Tüm üstler |
| 4 | Multiplicative | `a*b`, `a/b`, `a%b` | Soldan sağa | Tüm üstler |
| 5 | Additive | `a+b`, `a-b` | Soldan sağa | Tüm üstler |
| 6 | Shift | `a<<b`, `a>>b` | Parantez gerektirir | Unary |
| 7 | Relational | `a<b`, `a>b`, `a<=b`, `a>=b`, `a==b`, `a!=b` | Parantez gerektirir | Tüm üstler |
| 8 | Binary AND | `a&b` | Soldan sağa | Unary |
| 9 | Binary XOR | `a^b` | Soldan sağa | Unary |
| 10 | Binary OR | `a\|b` | Soldan sağa | Unary |
| 11 | Short-circuit AND | `a&&b` | Soldan sağa | Relational |
| 12 | Short-circuit OR | `a\|\|b` | Soldan sağa | Relational |

**Önemli kısıtlamalar:**
- **Birbirleriyle ilişkilendirilemeyen gruplar:** Short-circuit OR/AND, Binary OR/AND/XOR. Bu grupları bir arada kullanmak için parantez gerekir.
- **Kendileriyle ilişkilendirilemeyen gruplar:** Shift ve Relational. Bunları zincirleme kullanmak için parantez gerekir.

```wgsl
// Geçersiz ifadeler (parantez gerektirir):
// x & y ^ z | w      → let a = x & (y ^ (z | w));
// x + y << z >= w     → let b = (x + y) << (z >= w);
// x < y > z           → let c = x < (y > z);
// x && y || z         → let d = x && (y || z);
```

---

> **Önceki:** [← Tip Sistemi](02-tip-sistemi.md) · **Sonraki:** [Program Akışı ve Fonksiyonlar →](04-program-akisi-ve-fonksiyonlar.md)
