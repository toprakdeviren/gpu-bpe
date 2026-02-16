---
title: Program Akışı ve Fonksiyonlar
section: 9-11
source: "W3C WGSL Spec §9–§11"
---

# 4. Program Akışı ve Fonksiyonlar

> Kontrol yapıları, davranış analizi, assertion mekanizması, fonksiyonlar ve alias analizi.

---

## §9 Statements

**Statement** (ifade), programın yürütülmesini kontrol eden bir program parçasıdır. Statement'lar genellikle sıralı olarak çalıştırılır; ancak **kontrol akışı** (control flow) statement'ları programın sıralı olmayan şekilde yürütülmesine neden olabilir.

### 9.1 Compound Statement

**Compound statement**, süslü parantezlerle `{ }` çevrili sıfır veya daha fazla statement'tan oluşan bir bloktur.

```wgsl
{
  var x: i32 = 0;
  x = x + 1;
  // x'in kapsamı bu blok sonunda biter
}
```

**Kapsam kuralı:** Bir compound statement içinde yapılan bildirim (declaration), bir sonraki statement'ın başlangıcından compound statement'ın sonuna kadar **kapsam dahilindedir** (in scope).

> **Not:** `continuing_compound_statement`, `loop` statement'ının gövdesinin sonunda yer alan ve isteğe bağlı bir `break if` ifadesine izin veren özel bir compound statement formudur.

### 9.2 Assignment Statement

Assignment (atama) ifadesi, bir ifadeyi değerlendirir ve isteğe bağlı olarak sonucu belleğe yazar (bir değişkenin içeriğini günceller).

Operatörün solundaki kısım **sol taraf** (left-hand side), sağındaki ifade ise **sağ taraf** (right-hand side) olarak adlandırılır.

#### 9.2.1 Simple Assignment

Basit atama, sol tarafın bir ifade ve operatörün `=` olduğu durumdur. Sağ tarafın değeri, sol tarafın referans verdiği bellek konumuna yazılır.

**Ön koşullar:**
- `e: T` — `T` concrete constructible tip olmalı
- `r: ref<AS, T, AM>` — `AS` yazılabilir adres uzayı, `AM` `write` veya `read_write` olmalı

```wgsl
struct S {
  age: i32,
  weight: f32
}
var<private> person: S;

fn f() {
  var a: i32 = 20;
  a = 30;                           // a'nın içeriğini 30 ile değiştir

  person.age = 31;                  // person yapısının age alanına yaz

  var uv: vec2<f32>;
  uv.y = 1.25;                      // uv'nin ikinci bileşenine yaz

  let uv_x_ptr: ptr<function, f32> = &uv.x;
  *uv_x_ptr = 2.5;                  // Pointer üzerinden yazma

  var sibling: S;
  sibling = person;                 // Struct kopyalama
}
```

> **Not:** Referans **invalid memory reference** ise, yazma işlemi gerçekleşmeyebilir veya beklenmeyen bir bellek konumuna yazılabilir.

#### 9.2.2 Phony Assignment

**Phony assignment**, sol tarafın `_` (underscore) token'ı olduğu durumdur. Sağ taraf değerlendirilir ancak sonucu **saklanmaz**.

```wgsl
_ = e     // e değerlendirilir, sonucu atılır
```

`_` bir identifier değildir, bu yüzden bir ifadede kullanılamaz.

**Kullanım alanları:**

1. **Dönüş değerini atmak:** Değer döndüren bir fonksiyonu çağırıp sonucu kullanmamak:

```wgsl
var<private> counter: i32;

fn increment_and_yield_previous() -> i32 {
  let previous = counter;
  counter = counter + 1;
  return previous;
}

fn user() {
  // Sayacı artır ama sonucu kullanma
  _ = increment_and_yield_previous();
}
```

2. **Kaynak bağlamalarını aktif etmek:** Bir değişkeni statik erişimle (statically accessed) shader'ın resource interface'ine dahil etmek:

```wgsl
struct BufferContents {
  counter: atomic<u32>,
  data: array<vec4<f32>>
}
@group(0) @binding(0) var<storage> buf: BufferContents;
@group(0) @binding(1) var t: texture_2d<f32>;
@group(0) @binding(2) var s: sampler;

@fragment
fn shade_it() -> @location(0) vec4<f32> {
  // buf, t ve s'yi shader arayüzünün parçası olarak bildir
  _ = &buf;    // Constructible olmayan tipler için pointer kullan
  _ = t;
  _ = s;
  return vec4<f32>();
}
```

#### 9.2.3 Compound Assignment

**Compound assignment**, işlem ve atamayı birleştiren kısa yol operatörleridir.

| Statement | Expansion | Açıklama |
|-----------|-----------|----------|
| `e1 += e2` | `e1 = e1 + (e2)` | Toplama ve atama |
| `e1 -= e2` | `e1 = e1 - (e2)` | Çıkarma ve atama |
| `e1 *= e2` | `e1 = e1 * (e2)` | Çarpma ve atama |
| `e1 /= e2` | `e1 = e1 / (e2)` | Bölme ve atama |
| `e1 %= e2` | `e1 = e1 % (e2)` | Kalan ve atama |
| `e1 &= e2` | `e1 = e1 & (e2)` | Bitwise AND ve atama |
| `e1 \|= e2` | `e1 = e1 \| (e2)` | Bitwise OR ve atama |
| `e1 ^= e2` | `e1 = e1 ^ (e2)` | Bitwise XOR ve atama |
| `e1 >>= e2` | `e1 = e1 >> (e2)` | Sağa kaydırma ve atama |
| `e1 <<= e2` | `e1 = e1 << (e2)` | Sola kaydırma ve atama |

**Önemli kurallar:**
- Referans ifadesi `e1` **yalnızca bir kez** değerlendirilir.
- `e1` için referans tipinin erişim modu `read_write` olmalıdır.
- Compound assignment, phony assignment ile birleştirilemez (`_ += e` geçersizdir).
- Sağ taraf bağımsız bir ifade olarak ayrıştırılır: `value *= 2 + 3` aslında `value = value * (2 + 3)` demektir.

```wgsl
var<private> next_item: i32 = 0;

fn advance_item() -> i32 {
   next_item += 1;     // next_item'a 1 ekle
   return next_item - 1;
}

fn bump_item() {
  var data: array<f32, 10>;
  next_item = 0;
  // data[0]'a 5.0 ekle, advance_item() yalnızca BİR KEZ çağrılır
  data[advance_item()] += 5.0;
  // next_item burada 1 olacaktır
}

fn precedence_example() {
  var value = 1;
  value *= 2 + 3;     // value = value * (2 + 3) = 5
}
```

> **Not:** Referans `e1` bir kez değerlendirilse de, altta yatan belleğe **iki kez** erişilir: önce eski değeri okumak için read access, sonra güncel değeri yazmak için write access.

### 9.3 Increment and Decrement Statements

**Increment** (`++`) bir değişkenin içeriğine 1 ekler, **decrement** (`--`) 1 çıkarır.

**Ön koşullar:**
- İfade, concrete integer scalar (`i32` veya `u32`) store type'a ve `read_write` erişim moduna sahip bir referansa çözümlenmelidir.

| Statement | Eşdeğer | Açıklama |
|-----------|---------|----------|
| `r++` | `r += T(1)` | Bellek içeriğine 1 ekle |
| `r--` | `r -= T(1)` | Bellek içeriğinden 1 çıkar |

```wgsl
fn f() {
  var a: i32 = 20;
  a++;        // a = 21
  a--;        // a = 20
}
```

> **Not:** `++` ve `--` ifade (expression) değil, **statement**'tır. `let b = a++` gibi kullanımlar WGSL'de **geçersizdir**. C/C++'daki gibi prefix/postfix ayrımı yoktur.

### 9.4 Control Flow

Kontrol akışı statement'ları, programın sıralı olmayan şekilde yürütülmesine neden olabilir.

#### 9.4.1 If Statement

`if` ifadesi, koşul ifadelerinin değerlendirilmesine bağlı olarak en fazla **bir** compound statement'ı koşullu olarak çalıştırır.

**Yapı:** Bir `if` cümlesi, ardından sıfır veya daha fazla `else if` cümlesi, ardından isteğe bağlı bir `else` cümlesi.

**Tip kuralı:** Her `if` ve `else if` cümlesindeki ifade `bool` tipinde olmalıdır.

**Yürütme akışı:**
1. `if` koşulu değerlendirilir. `true` ise ilk compound statement çalışır.
2. Değilse, sıradaki `else if` koşulları sırayla değerlendirilir; ilk `true` olan çalışır.
3. Hiçbir koşul `true` değilse ve `else` cümlesi varsa, onun gövdesi çalışır.

```wgsl
if condition {
  // condition true ise
} else if other_condition {
  // other_condition true ise
} else {
  // hiçbiri değilse
}
```

#### 9.4.2 Switch Statement

`switch` ifadesi, bir seçici (selector) ifadesinin değerlendirmesine bağlı olarak kontrolü bir dizi `case` cümlesinden birine veya `default` cümlesine aktarır.

**Kurallar:**
- Her switch ifadesinde **tam olarak bir** `default` cümlesi olmalıdır.
- Selector ifadesi ve tüm case selector ifadeleri **aynı concrete integer scalar** tipinde olmalıdır.
- Case selector ifadeleri **const-expression** olmalıdır.
- Aynı switch'te iki farklı case selector ifadesi **aynı değere** sahip olmamalıdır.
- Case gövdesinin sonuna ulaşıldığında, kontrol switch ifadesinden sonraki ilk statement'a aktarılır (C'deki gibi **fall-through** yoktur).

```wgsl
var a: i32;
let x: i32 = generateValue();
switch x {
  case 0: {                      // İki nokta isteğe bağlı
    a = 1;
  }
  default {                      // default son olmak zorunda değil
    a = 2;
  }
  case 1, 2, {                   // Birden fazla seçici değer; sondaki virgül isteğe bağlı
    a = 3;
  }
  case 3 {
    a = 4;
  }
}
```

```wgsl
// default, başka case'lerle birleştirilebilir
const c = 2;
switch x {
  case 0: { a = 1; }
  case 1, c { a = 3; }          // const-expression case seçicisi olarak kullanılabilir
  case 3, default { a = 4; }    // default anahtar kelimesi diğer case'lerle birleştirilebilir
}
```

#### 9.4.3 Loop Statement

`loop` ifadesi, bir **loop body**'yi (döngü gövdesi) tekrar tekrar çalıştırır. Her çalıştırma bir **iterasyon** (iteration) olarak adlandırılır.

Bu tekrarlama, bir `break` veya `return` ifadesiyle kesilebilir.

İsteğe bağlı olarak, loop gövdesinin son ifadesi bir `continuing` ifadesi olabilir.

```wgsl
// Temel loop yapısı
var a: i32 = 2;
var i: i32 = 0;
loop {
  if i >= 4 { break; }

  a = a * 2;

  i++;
}
// a = 32
```

```wgsl
// Loop + continuing + break if
var a: i32 = 2;
var i: i32 = 0;
loop {
  let step: i32 = 1;

  if i % 2 == 0 { continue; }

  a = a * 2;

  continuing {
    i = i + step;
    break if i >= 4;     // continuing içinde koşullu break
  }
}
```

**Önemli kurallar:**
- Sınırsız sayıda iterasyon çalıştırılırsa **dynamic error** oluşur. Bu, döngünün erken sonlandırılmasına veya **device loss**'a yol açabilir.
- Loop gövdesindeki bildirimler her iterasyonda yeniden oluşturulur ve yeniden başlatılır.

> **Not:** `loop` ifadesi WGSL'e özgü bir yapıdır. Çoğu durumda `for` veya `while` tercih edilmelidir. `loop`, derlenmiş kodda yaygın bulunan döngü kalıplarını doğrudan ifade eder.

#### 9.4.4 For Statement

`for` ifadesi, bir `loop` ifadesinin üzerine **syntactic sugar** (sözdizimsel kolaylık) sağlar.

**Genel form:**
```
for (initializer ; condition ; update_part) { body }
```

**Eşdeğer loop dönüşümü (condition varsa):**
```
{
  initializer;
  loop {
    if !(condition) { break; }
    body
    continuing { update_part }
  }
}
```

**Kurallar:**
- **Initializer:** Döngüden önce bir kez çalıştırılır. Bildirilen tanımlayıcı, döngü gövdesinin sonuna kadar kapsam dahilindedir. Her iterasyonda **yeniden başlatılmaz.**
- **Condition:** Bool tipinde olmalıdır. Her iterasyonun başında değerlendirilir; `false` ise döngüden çıkılır.
- **Update part:** Her iterasyonun sonunda continuing ifadesi olarak çalışır.
- **Body:** Özel bir compound statement formu. Gövde içindeki bildirimler her iterasyonda yeniden oluşturulur.

```wgsl
var a: i32 = 2;
for (var i: i32 = 0; i < 4; i++) {
  a *= 2;
}
// a = 32

// Koşul olmadan da kullanılabilir (sonsuz döngü):
for (var i: i32 = 0; ; i++) {
  if i == 4 { break; }
  a = a + 2;
}
```

#### 9.4.5 While Statement

`while` ifadesi, bir koşula bağlı döngüdür. Her iterasyonun başında boolean koşul değerlendirilir; `false` ise döngü sona erer.

**Koşul `bool` tipinde olmalıdır.**

Aşağıdaki üç form birbirine eşdeğerdir:
- `while condition { body }`
- `loop { if !condition { break; } body }`
- `for (; condition ;) { body }`

```wgsl
var i: i32 = 0;
while i < 10 {
  // ...
  i++;
}
```

- Sınırsız sayıda iterasyon → **dynamic error** (device loss olabilir).

#### 9.4.6 Break Statement

`break` ifadesi, kontrolü en yakın çevreleyen döngü veya `switch` ifadesinin hemen sonrasına aktarır, böylece o yapının yürütülmesini sonlandırır.

**Kurallar:**
- Yalnızca `loop`, `for`, `while` ve `switch` ifadeleri içinde kullanılabilir.
- `continuing` bloğundan döngüyü sonlandırmak için `break` **kullanılamaz**. Bunun yerine `break if` kullanılmalıdır.

```wgsl
loop {
  // ...
  continuing {
    if i >= 4 { break; }    // ❌ Geçersiz! continuing içinde break olmaz
  }
}
```

#### 9.4.7 Break-If Statement

`break if` ifadesi bir boolean koşul değerlendirir; koşul `true` ise en yakın çevreleyen `loop`'un hemen sonrasına kontrolü aktarır.

**Kurallar:**
- Koşul `bool` tipinde olmalıdır.
- Yalnızca bir `continuing` bloğunun **son ifadesi** olarak kullanılabilir.

```wgsl
var a: i32 = 2;
var i: i32 = 0;
loop {
  let step: i32 = 1;

  if i % 2 == 0 { continue; }

  a = a * 2;

  continuing {
    i = i + step;
    break if i >= 4;        // ✅ Geçerli: continuing'in son ifadesi
  }
}
```

#### 9.4.8 Continue Statement

`continue` ifadesi, en yakın çevreleyen döngüde kontrolü aktarır:
- Eğer bir `continuing` bloğu varsa → oraya atlar (ileri dallanma).
- Yoksa → döngü gövdesinin başına geri döner (sonraki iterasyon).

**Kurallar:**
- Yalnızca `loop`, `for` ve `while` içinde kullanılabilir.
- Çevreleyen bir `continuing` bloğuna kontrol akışı **aktaramaz** (kendi continuing'ine doğrudan atlamak yasak).
- Hedeflenen `continuing` bloğunda kullanılan bir bildirimden **sonra atlanamaz** (o bildirim bypass edilemez).

```wgsl
// ❌ Geçersiz: continue, step bildirimini atlıyor ama continuing bunu kullanıyor
var i: i32 = 0;
loop {
  if i >= 4 { break; }
  if i % 2 == 0 { continue; }     // step'in tanımını atlıyor

  let step: i32 = 2;

  continuing {
    i = i + step;                  // step burada kullanılıyor → hata!
  }
}
```

#### 9.4.9 Continuing Statement

`continuing` ifadesi, bir döngü iterasyonunun sonunda çalıştırılacak bir compound statement belirtir. İsteğe bağlıdır.

**Kısıtlama:** `continuing` bloğu içinde hiçbir iç içe geçme seviyesinde `return` ifadesi **kullanılamaz**.

```wgsl
loop {
  // loop body
  continuing {
    // Her iterasyonun sonunda çalışır
    // (continue veya normal akış ile ulaşılır)
    // NOT: Burada return kullanılamaz!
  }
}
```

#### 9.4.10 Return Statement

`return` ifadesi, mevcut fonksiyonun yürütülmesini sonlandırır.

- Fonksiyon bir **entry point** ise, shader invocation'ı sonlandırılır.
- Değilse, çağrı noktasından (call site) sonraki ifade/statement'a devam edilir.

**Kurallar:**
- Fonksiyonun dönüş tipi yoksa: `return` isteğe bağlıdır. Sağlanırsa değer içermemelidir.
- Fonksiyonun dönüş tipi varsa: `return` ifadesi **zorunludur** ve dönüş değeri fonksiyonun dönüş tipiyle eşleşmelidir.

```wgsl
fn add(a: f32, b: f32) -> f32 {
  return a + b;               // Dönüş değeri zorunlu
}

fn do_something() {
  // ...
  return;                     // Dönüş değeri yok (isteğe bağlı)
}
```

#### 9.4.11 Discard Statement

`discard` ifadesi, mevcut invocation'ı bir **helper invocation**'a dönüştürür ve fragment'ı atar. Yalnızca **fragment shader** aşamasında kullanılabilir.

**Etkiler:**
- Invocation, helper invocation'a dönüştürülür.
- Mevcut fragment, GPU render pipeline'ında daha ileri işlenmez.
- Yalnızca `discard`'dan **önce** çalıştırılan statement'lar gözlemlenebilir etkilere sahip olacaktır.

```wgsl
@group(0) @binding(0)
var<storage, read_write> will_emit_color: u32;

fn discard_if_shallow(pos: vec4<f32>) {
  if pos.z < 0.001 {
    // Bu çalıştırılırsa, will_emit_color asla 1 olarak ayarlanmaz
    // çünkü helper invocation'lar paylaşımlı belleğe yazmaz
    discard;
  }
  will_emit_color = 1;
}

@fragment
fn main(@builtin(position) coord_in: vec4<f32>)
  -> @location(0) vec4<f32>
{
  discard_if_shallow(coord_in);
  will_emit_color = 1;
  return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
```

> **Not:** `discard`, fragment aşamasındaki herhangi bir fonksiyonda çalıştırılabilir (yalnızca entry point'te değil) ve etki aynıdır: fragment atılır.

### 9.5 Function Call Statement

Fonksiyon çağrı ifadesi, bir fonksiyon çağrısını çalıştırır. Dönüş değeri olan bir fonksiyon çağrılıyorsa ve fonksiyonda `@must_use` attribute'u varsa, sonucu atmak **shader-creation error** oluşturur.

```wgsl
do_work();                    // Dönüş tipi olmayan fonksiyon çağrısı
_ = compute_value();          // Dönüş değeri açıkça atılır (must_use yoksa)
```

> **Not:** Fonksiyon bir değer döndürüyor ve `@must_use` attribute'u yoksa, dönüş değeri sessizce yok sayılır.

### 9.6 Statements Grammar Summary

```
statement :
  | ';'                                // Boş statement
  | return_statement ';'
  | if_statement
  | switch_statement
  | loop_statement
  | for_statement
  | while_statement
  | func_call_statement ';'
  | variable_or_value_statement ';'
  | break_statement ';'
  | continue_statement ';'
  | 'discard' ';'
  | variable_updating_statement ';'
  | compound_statement
  | assert_statement ';'

variable_updating_statement :
  | assignment_statement
  | increment_statement
  | decrement_statement
```

**Özel bağlam gerektiren statement'lar:**
- `break_if_statement` — yalnızca `continuing` bloğunun son ifadesi olarak
- `continuing_compound_statement` — yalnızca `loop` gövdesinin son ifadesi olarak

### 9.7 Statements Behavior Analysis

Behavior analysis (davranış analizi), her statement'ın yürütülmesi tamamlandıktan sonra kontrolün **nasıl devam edeceğini** özetleyen bir sistemdir. Bu analiz, hem kontrol akışı kurallarının geçerliliğini doğrulamak hem de **uniformity analysis** (§15.2) için kullanılır.

#### 9.7.1 Rules

Bir **behavior** (davranış), aşağıdaki elemanlardan oluşabilen bir kümedir:

| Behavior Elemanı | Anlamı |
|-------------------|--------|
| **Return** | Fonksiyon, `return` ile sonlanır |
| **Break** | Döngü/switch, `break` ile sonlanır |
| **Continue** | Döngü, `continue` ile sonraki iterasyona geçer |
| **Next** | Bir sonraki statement'a düşer (fall-through) |

**Fonksiyon kuralları:**
- Dönüş tipi olan fonksiyonun gövdesi **{Return}** behavior'una sahip olmalıdır (tüm yollarda return olmalı).
- Dönüş tipi olmayan fonksiyonun gövdesi **{Next, Return}**'ün bir alt kümesi olmalıdır.

**Temel statement behavior'ları:**

| Statement | Sonuç Behavior |
|-----------|---------------|
| Boş statement (`;`) | {Next} |
| `break;` | {Break} |
| `continue;` | {Continue} |
| `return;` / `return e;` | {Return} |
| `discard;` | {} |
| Atama, increment, decrement | {Next} |
| Fonksiyon çağrısı | Fonksiyonun behavior'u |

**Bileşik behavior kuralları:**

- **Sıralı compound:** `{ s1; s2; }` → İlk statement'ın Next olmayan behavior'ları + ikinci statement'ın behavior'u.
- **if/else:** Tüm dalların behavior birleşimi. `else` yoksa {Next} eklenir.
- **switch:** Tüm case'lerin behavior birleşimi (Break → Next'e dönüştürülür).
- **loop:** Body behavior'undaki Continue → Next'e, Break → Next'e dönüştürülür. Next kaldırılır (sonsuz döngü anlamına gelir).
- **for/while:** Loop behavior'una benzer, ancak koşul false olabilir → {Next} eklenir.

#### 9.7.2 Notes

- Behavior analizi **statik analiz** olduğundan, dead code (erişilemeyen kod) derleme hataları üretebilir.
- Bir fonksiyonun behavior'u, gövde behavior'undaki her "Return" yerine "Next" konularak hesaplanır.
- Sonuç olarak bir fonksiyonun behavior'u her zaman `{}` (hiç dönmez) veya `{Next}` (normal dönüş) olur.

#### 9.7.3 Examples

```wgsl
// ✅ Behavior: {Return} — dönüş tipi olan fonksiyon için geçerli
fn valid_return(x: i32) -> i32 {
  if x > 0 {
    return x;                  // {Return}
  } else {
    return -x;                 // {Return}
  }
  // if/else behavior: {Return} ∪ {Return} = {Return}
}

// ❌ Hata: Behavior {Next, Return} — tüm yollarda return yok
fn invalid_return(x: i32) -> i32 {
  if x > 0 {
    return x;                  // {Return}
  }
  // else yok → {Next} eklenir
  // Sonuç: {Next, Return} — dönüş tipi olan fonksiyon için geçersiz!
}

// ✅ Behavior: {Next, Return} — dönüş tipi olmayan fonksiyon için geçerli
fn valid_void(x: i32) {
  if x > 0 {
    return;                    // {Return}
  }
  // {Next} — sorun yok
}
```

---

## §10 Assertions

**Assertion** (doğrulama), bir boolean koşulun sağlandığını garanti eden bir kontroldür.

### 10.1 Const Assertion Statement

`const_assert` ifadesi, koşul `false` olarak değerlendirilirse **shader-creation error** üreten bir doğrulamadır.

**Kurallar:**
- İfade `bool` tipinde olmalıdır.
- İfade bir **const-expression** olmalıdır.
- Hem **module scope** hem de **function scope**'ta kullanılabilir.
- Derlenmiş shader üzerinde hiçbir etkisi yoktur (yalnızca derleme zamanı kontrolü).

```wgsl
const WORKGROUP_SIZE = 256;
const MAX_SIZE = 1024;

// Module scope'ta kullanım
const_assert WORKGROUP_SIZE <= MAX_SIZE;
const_assert WORKGROUP_SIZE % 32 == 0;

fn foo() {
  const x = 1;
  const y = 2;
  const z = x + y - 2;

  const_assert z > 0;                 // ✅ Geçerli: z const-expression

  let a = 3;
  // const_assert a != 0;             // ❌ Geçersiz: a const-expression değil (let)
}
```

**Parantezler isteğe bağlıdır:**
```wgsl
const_assert x < y;         // Parantez yok
const_assert(y != 0);       // Parantez ile
```

**Kullanım senaryoları:**
- Derleme zamanında yapılandırma parametrelerini doğrulama
- Workgroup boyutlarının sınırlar dahilinde olduğunu kontrol etme
- Template/const parametrelerinin tutarlılığını garanti etme

---

## §11 Functions

**Fonksiyon**, çağrıldığında hesaplama işi gerçekleştiren bir birimdir. WGSL'de iki tür fonksiyon vardır:

1. **Built-in fonksiyonlar:** WGSL implementasyonu tarafından sağlanır, her zaman kullanılabilir (§17).
2. **Kullanıcı tanımlı fonksiyonlar:** WGSL modülünde bildirilir.

**Çağırma yolları:**
- Fonksiyon çağrı ifadesi ile (§8.10 — dönüş değeri olan)
- Fonksiyon çağrı statement'ı ile (§9.5 — dönüş değeri yok sayılan)
- Entry point olarak WebGPU implementasyonu tarafından (§13)

> **Not:** Fonksiyonlar kaynak kodda herhangi bir sırada tanımlanabilir; forward declaration gerekmez.

### 11.1 Declaring a User-defined Function

Fonksiyon bildirimi aşağıdaki unsurları belirtir:

- İsteğe bağlı **attribute**'lar (ör. `@vertex`, `@fragment`, `@compute`, `@workgroup_size`)
- Fonksiyonun **adı**
- Sıralı **formal parametre** listesi (virgülle ayrılmış, parantez içinde)
- İsteğe bağlı **dönüş tipi** (attribute'larla birlikte)
- **Fonksiyon gövdesi** (çağrıldığında çalıştırılacak statement'lar)

**Kurallar:**
- Fonksiyon bildirimleri yalnızca **module scope**'ta yapılabilir.
- Fonksiyon adı tüm program boyunca kapsam dahilindedir.
- Her kullanıcı tanımlı fonksiyonun yalnızca **bir overload'u** vardır.
- Dönüş tipi belirtilirse, **constructible** tip olmalıdır.
- Parametrelerin tipleri: constructible, pointer, texture veya sampler olabilir.
- İki formal parametre aynı ada sahip olmamalıdır.

```wgsl
// Basit fonksiyon: iki parametre (i32 ve f32), i32 döndürür
fn add_two(i: i32, b: f32) -> i32 {
  return i + 2;     // Formal parametre gövdede kullanılabilir
}

// Compute shader entry point
@compute @workgroup_size(1)
fn main() {
   let six: i32 = add_two(4, 5.0);
}
```

**Parametre ve dönüş tipi attribute'ları:**

| Attribute | Uygulama Alanı | Açıklama |
|-----------|---------------|----------|
| `@builtin` | Parametre / Dönüş | Built-in değer (ör. `position`, `global_invocation_id`) |
| `@location` | Parametre / Dönüş | Inter-stage veya output location |
| `@blend_src` | Dönüş | Dual-source blending kaynağı |
| `@interpolate` | Parametre / Dönüş | Interpolasyon modu |
| `@invariant` | Dönüş | Pozisyon çıkışı değişmezliği |

**Fonksiyon attribute'ları:**

| Attribute | Açıklama |
|-----------|----------|
| `@vertex` | Vertex shader entry point |
| `@fragment` | Fragment shader entry point |
| `@compute` | Compute shader entry point |
| `@workgroup_size(x, y, z)` | Compute shader workgroup boyutu |

### 11.2 Function Calls

Fonksiyon çağrısı, bir fonksiyonu çalıştıran bir statement veya ifadedir.

**Terminoloji:**
- **Calling function (caller):** Çağrıyı yapan fonksiyon
- **Called function (callee):** Çağrılan fonksiyon
- **Call site:** Çağrının kaynak koddaki konumu

**Çağrı kuralları:**
- Argüman sayısı, formal parametre sayısıyla eşleşmelidir.
- Her argüman tipi, karşılık gelen parametrenin tipiyle uyumlu olmalıdır.
- Argüman değerlendirme sırası: **soldan sağa**.

**Çağrı yürütme adımları:**

1. Argüman değerleri soldan sağa değerlendirilir.
2. Çağıran fonksiyon askıya alınır (tüm yerel değişkenler ve sabitler durumlarını korur).
3. Çağrılan fonksiyon kullanıcı tanımlı ise, function scope değişkenleri için bellek ayrılır ve başlatılır.
4. Formal parametrelere, çağrı argümanları pozisyona göre eşlenir.
5. Kontrol çağrılan fonksiyona aktarılır (gövdenin ilk statement'ından başlar).
6. Çağrılan fonksiyon dönene kadar çalıştırılır.
7. Kontrol çağıran fonksiyona geri aktarılır. Dönüş değeri varsa, çağrı ifadesinin değeri olarak sağlanır.

**Dönüş koşulları:**
- **Built-in:** İşi tamamlandığında döner.
- **Dönüş tipli kullanıcı tanımlı:** `return` statement'ı çalıştırıldığında.
- **Dönüş tipi olmayan kullanıcı tanımlı:** `return` veya gövdenin sonuna ulaşıldığında.

> **Not:** Fragment shader'da bir çağrıdaki tüm quad invocation'lar discard edilmişse, fonksiyon çağrısı asla **dönmeyebilir**.

### 11.3 `const` Functions

`@const` attribute'u ile bildirilen bir fonksiyon, **shader-creation zamanında** (derleme zamanı) değerlendirilebilir. Bu fonksiyonlara **const-function** denir.

- Const-function çağrıları, **const-expression**'ların parçası olabilir.
- Fonksiyonun gövdesindeki tüm ifadeler const-expression olmalı ve tüm bildirimler const-declaration olmalıdır.

> **Önemli:** `@const` attribute'u **kullanıcı tanımlı** fonksiyonlara uygulanamaz. Yalnızca built-in fonksiyonlar const-function olabilir.

```wgsl
// firstLeadingBit bir const-function'dır (built-in)
const first_one = firstLeadingBit(1234 + 4567);  // Değer: 12, Tip: i32

@id(1) override x: i32;
override y = firstLeadingBit(x);  // override-expression olarak kullanılabilir
                                   // (bu bağlamda const-expression DEĞİL)

fn foo() {
  // Dizi boyutu olarak const-expression'da kullanılabilir
  var a: array<i32, firstLeadingBit(257)>;
}
```

### 11.4 Restrictions on Functions

WGSL fonksiyonları üzerinde katı kısıtlamalar uygulanır:

| Kısıtlama | Açıklama |
|-----------|----------|
| **Vertex shader** | `position` built-in output value döndürmelidir |
| **Entry point** | Fonksiyon çağrısının hedefi olamaz (yalnızca WebGPU tarafından çağrılır) |
| **Dönüş tipi** | Constructible tip olmalıdır |
| **Parametre tipi** | Constructible, pointer, texture veya sampler olmalıdır |
| **Argüman tipi** | Her argüman, karşılık gelen parametrenin tipiyle eşleşmelidir |
| **Pointer argüman** | Address space, store type ve access mode eşleşmelidir |
| **Rekürsiyon** | ❌ Bildirimler arasında döngüsel bağımlılıklar yasaktır |

```wgsl
fn bar(p: ptr<function, f32>) { }
fn baz(p: ptr<private, i32>) { }
fn baz2(p: ptr<storage, f32>) { }

@group(0) @binding(0) var<storage> ro_storage: f32;
@group(0) @binding(1) var<storage, read_write> rw_storage: f32;

fn foo() {
  var usable_func: f32;
  var i32_func: i32;

  bar(&usable_func);              // ✅ Geçerli: function address space, f32
  baz(&i32_func);                 // ❌ Geçersiz: address space uyumsuzluğu (function vs private)
  baz2(&ro_storage);              // ✅ Geçerli: storage, read, f32
  baz2(&rw_storage);              // ❌ Geçersiz: access mode uyumsuzluğu (read_write vs read)
}
```

> **Not:** Rekürsiyon, bildirimler arasında döngüsel bağımlılıklara izin verilmediği için yasaktır.

#### 11.4.1 Alias Analysis

WGSL, **alias analizi** kurallarıyla yazma-yazma ve okuma-yazma çakışmalarını önler. Bu analiz, aynı bellek konumuna birden fazla yoldan erişimi kısıtlar.

##### 11.4.1.1 Root Identifier

Bir fonksiyon içinde her **memory view** (referans veya pointer) belirli bir **root identifier**'a sahiptir. Root identifier, o belleğe ilk erişimi sağlayan değişken veya pointer parametresidir.

**Root identifier belirleme kuralları:**

| İfade formu | Root identifier |
|-------------|----------------|
| Değişken adı `v` | `v` kendisi |
| Pointer parametresi `p` | `p` kendisi |
| `let x = E2` → `x` kullanımı | `E2`'nin root identifier'ı |
| `(E2)`, `&E2`, `*E2`, `E2[i]` | `E2`'nin root identifier'ı |
| `E2.member`, `E2.swizzle` | `E2`'nin root identifier'ı |

##### 11.4.1.2 Aliasing

İki root identifier, aynı **originating variable**'a sahip olduğunda **alias** oluştururlar.

**Kural:** Bir WGSL fonksiyonunun yürütülmesi, alias olan root identifier'lar aracılığıyla belleğe potansiyel erişim **yapmamalıdır** — eğer bu erişimlerden biri yazma ve diğeri okuma veya yazma ise.

**Call site kuralları:** Her fonksiyon çağrısında aşağıdakiler **shader-creation error** oluşturur:

1. Aynı root identifier'a sahip iki pointer argümanı ve karşılık gelen parametrelerden biri yazma kümesindeyse.
2. Root identifier'ı module-scope değişkenine eşlenen bir pointer argümanı ve karşılık gelen parametre yazma kümesindeyken, aynı module-scope değişkeni çağrılan fonksiyonda okunuyorsa.
3. Root identifier'ı module-scope değişkenine eşlenen bir pointer argümanı ve karşılık gelen parametre okuma kümesindeyken, aynı module-scope değişkeni çağrılan fonksiyonda yazılıyorsa.

```wgsl
var<private> x: i32 = 0;

fn f1(p1: ptr<function, i32>, p2: ptr<function, i32>) {
  *p1 = *p2;                  // p1 yazılır, p2 okunur
}

fn f4(p1: ptr<function, i32>, p2: ptr<function, i32>) -> i32 {
  return *p1 + *p2;           // Her ikisi de sadece okunur
}

fn f6(p: ptr<private, i32>) {
  x = *p;                     // x yazılır (global), p okunur
}

fn f7(p: ptr<private, i32>) -> i32 {
  return x + *p;              // İkisi de sadece okunur
}

fn f3() {
  var a: i32 = 0;
  f1(&a, &a);                 // ❌ Geçersiz: aynı root identifier, biri yazılıyor
}

fn f5() {
  var a: i32 = 0;
  let b = f4(&a, &a);         // ✅ Geçerli: her ikisi de sadece okunuyor
}

fn f8() {
  let a = f6(&x);             // ❌ Geçersiz: x hem global yazma hem parametre okuma
  let b = f7(&x);             // ✅ Geçerli: x yalnızca okunuyor (global + parametre)
}
```

---

> **Önceki:** [← Değişkenler ve İfadeler](03-degiskenler-ve-ifadeler.md) · **Sonraki:** [GPU Arayüzü ve Bellek →](05-gpu-arayuzu-ve-bellek.md)
