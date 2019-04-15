#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h> /* memset */
#include <stdbool.h>
#include <stdio.h>
#include <time.h>


#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

#define CEILDIV(a,b) (a / b + (a % b != 0))

#define BITMAP_GET(bm,i) (bm[i / 64] & (1 << (i % 64)))
#define BITMAP_SET(bm,i) (bm[i / 64] |= ( 1 << (i % 64)))
#define BITMAP_CLR(bm,i) (bm[i / 64] &= ~(1 << (i % 64)))


/*
 * Myers' bit-parallel algorithm
 *
 * See: G. Myers. "A fast bit-vector algorithm for approximate string
 *      matching based on dynamic programming." Journal of the ACM, 1999.
 */
static uint64_t myers1999_geteq(uint64_t *Peq, uint64_t *map, uint64_t c)
{
    uint8_t h = c % 256;
    while (1) {
        if (map[h] == c)
            return Peq[h];
        if (map[h] == UINT64_MAX)
            return 0;
        h++;
    }
}

static void myers1999_setup(uint64_t *Peq, char *sb, uint64_t start, uint8_t len)
{
    uint64_t c;
    uint8_t h;

    memset(Peq, 0, sizeof(uint64_t) * 256);
    int i = len;
    while(i--) Peq[sb[i] & 0xFF] |= 1uL << i;

    // while (len--) {
    //     c = sb[start + len];
    //     h = c % 256;
    //     while (map[h] != UINT64_MAX && map[h] != c)
    //         h++;
    //     if (map[h] == UINT64_MAX) {
    //         map[h] = c;
    //         Peq[h] = 0;
    //     }
    //     Peq[h] |= (uint64_t) 1 << len;
    // }
}



#define PRECOMPUTE(Peq, sb, len)\
  i = len;\
  while(i--) Peq[sb[i] & 255] |= 1uL << i;\

static uint64_t myers1999_simple(char *sb1, char *sb2)
{
    uint8_t sb1len = strlen(sb1);
    uint8_t sb2len = strlen(sb2);
    uint64_t Peq[256] = {0};
    uint64_t Eq, Xv, Xh, Ph, Mh, Pv, Mv, Last;
    uint8_t i, Score;

    Mv = 0;
    Pv = -1;
    Score = sb2len;
    Last = 1uL << (sb2len - 1);

    PRECOMPUTE(Peq, sb2, sb2len);

    for (i = 0; i < sb1len; i++) {
        Eq = Peq[sb1[i] & 255];

        Xv = Eq | Mv;
        Xh = (Eq & Pv) + Pv ^ Pv | Eq;

        Ph = Mv | ~ (Xh | Pv);
        Mh = Pv & Xh;

        if (Ph & Last) ++Score;
        else if (Mh & Last) --Score;

        // if (Ph & Last) ++Score;
        // if (Mh & Last) --Score;

        Ph = Ph << 1 | 1;
        Pv = Mh << 1 | ~ (Xv | Ph);

        Mv = Ph & Xv;

    } 
    printf("%d\n", Score);
    printf("%lu\n", Mv);
    printf("%lu\n", Pv);



    return Score;
}






// static uint64_t myers1999_block(char*sb1, char *sb2, uint64_t b, uint64_t *Phc, uint64_t *Mhc)
// {
//     uint64_t sb1len = strlen(sb1);
//     uint64_t sb2len = strlen(sb2);
//     uint64_t Peq[256];
//     uint64_t map[256];
//     uint64_t Eq, Xv, Xh, Ph, Mh, Pv, Mv, Last;
//     uint8_t Pb, Mb, vlen;

//     uint64_t idx, start, Score;

//     start = b * 64;
//     vlen = MIN(64, sb2len - start);

//     Mv = 0;
//     Pv = -1;
//     Score = sb2len;
//     Last = (uint64_t) 1 << (vlen - 1);

//     myers1999_setup(Peq, sb2, start, vlen);

//     for (idx = 0; idx < sb1len; idx++) {
//         Eq = myers1999_geteq(Peq, map, sb1[idx]);

//         Pb = !!BITMAP_GET(Phc, idx);
//         Mb = !!BITMAP_GET(Mhc, idx);

//         Xv = Eq | Mv;
//         Eq |= Mb;
//         Xh = (((Eq & Pv) + Pv) ^ Pv) | Eq;

//         Ph = Mv | ~ (Xh | Pv);
//         Mh = Pv & Xh;

//         if (Ph & Last) {
//             BITMAP_SET(Phc, idx);
//             Score++;
//         } else {
//             BITMAP_CLR(Phc, idx);
//         }
//         if (Mh & Last) {
//             BITMAP_SET(Mhc, idx);
//             Score--;
//         } else {
//             BITMAP_CLR(Mhc, idx);
//         }

//         Ph = (Ph << 1) | Pb;
//         Mh = (Mh << 1) | Mb;

//         Pv = Mh | ~ (Xv | Ph);
//         Mv = Ph & Xv;
//     }
//     return Score;
// }

// static uint64_t myers1999(char *sb1, char *sb2)
// {
//     uint64_t sb1len = strlen(sb1);
//     uint64_t sb2len = strlen(sb2);
//     uint64_t i;
//     uint64_t vmax, hmax;
//     uint64_t *Phc, *Mhc;
//     uint64_t res;

//     if (sb2len == 0)
//         return sb1len;

//     if (sb2len <= 64)
//         return myers1999_simple(sb1, sb2);

//     hmax = CEILDIV(sb1len, 64);
//     vmax = CEILDIV(sb2len, 64);

//     Phc = malloc(hmax * sizeof(uint64_t));
//     Mhc = malloc(hmax * sizeof(uint64_t));


//     for (i = 0; i < hmax; i++) {
//         Mhc[i] = 0;
//         Phc[i] = ~ (uint64_t) 0;
//     }

//     for (i = 0; i < vmax; i++)
//         res = myers1999_block(sb1, sb2, i, Phc, Mhc);

//     free(Phc);
//     free(Mhc);

//     return res;
// }


// #include <stdint.h>
// #include <stdlib.h>
// #include <stdio.h>
// #include <string.h> /* memset */

// #include <stdio.h>
// #include <time.h>


// #define MIN(a,b) ((a) < (b) ? (a) : (b))
// #define MAX(a,b) ((a) > (b) ? (a) : (b))

// #define CEILDIV(a,b) (a / b + (a % b != 0))
// #define BITMAP_GET(bm,i) (bm[i / 64] & ((uint64_t) 1 << (i % 64)))
// #define BITMAP_SET(bm,i) (bm[i / 64] |= ((uint64_t) 1 << (i % 64)))
// #define BITMAP_CLR(bm,i) (bm[i / 64] &= ~((uint64_t) 1 << (i % 64)))


// /*
//  * Myers" bit-parallel algorithm
//  *
//  * See: G. Myers. "A fast bit-vector algorithm for approximate string
//  *      matching based on dynamic programming." Journal of the ACM, 1999.
//  */

// #define PRECOMPUTE(Peq, b, m)\
//   i = m;\
//   while(i--) Peq[b[i] & 0xFF] |= 1uL << i;


// static uint64_t myers1999_simple(char *a, char *b)
// {
//     uint64_t n = strlen(a);
//     uint64_t m = strlen(b);
//     uint64_t Peq[256] = {0};
//     uint64_t Eq, Xv, Xh, Ph, Mh, Pv, Mv, Score, Last, i;

//     Mv = 0;
//     Pv = -1;
//     Score = m;
//     Last = 1 << (m - 1);

//     PRECOMPUTE(Peq, b, m);

//     for (i = 0; i < n; i++) {
//         Eq = Peq[a[i] & 0xFF];

//         Xv = Eq | Mv;
//         Xh = (((Eq & Pv) + Pv) ^ Pv) | Eq;

//         Ph = Mv | ~ (Xh | Pv);
//         Mh = Pv & Xh;

//         if (Ph & Last)
//             Score += 1;
//         if (Mh & Last)
//             Score -= 1;

//         Ph = (Ph << 1) | 1;
//         Mh = (Mh << 1);

//         Pv = Mh | ~ (Xv | Ph);
//         Mv = Ph & Xv;
//     }
//     return Score;
// }

// static uint64_t myers1999_block(char*sb1, char *sb2, uint64_t b, uint64_t *Phc, uint64_t *Mhc)
// {
//     uint64_t sb1len = strlen(sb1);
//     uint64_t sb2len = strlen(sb2);
//     uint64_t Peq[256];
//     uint64_t map[256];
//     uint64_t Eq, Xv, Xh, Ph, Mh, Pv, Mv, Last;
//     uint8_t Pb, Mb, vlen;

//     uint64_t idx, start, Score;

//     start = b * 64;
//     vlen = MIN(64, sb2len - start);

//     Mv = 0;
//     Pv = ~ (uint64_t) 0;
//     Score = sb2len;
//     Last = (uint64_t) 1 << (vlen - 1);

//     myers1999_setup(Peq, map, sb2, start, vlen);

//     for (idx = 0; idx < sb1len; idx++) {
//         Eq = myers1999_geteq(Peq, map, sb1[idx]);

//         Pb = !!BITMAP_GET(Phc, idx);
//         Mb = !!BITMAP_GET(Mhc, idx);

//         Xv = Eq | Mv;
//         Eq |= Mb;
//         Xh = (((Eq & Pv) + Pv) ^ Pv) | Eq;

//         Ph = Mv | ~ (Xh | Pv);
//         Mh = Pv & Xh;

//         if (Ph & Last) {
//             BITMAP_SET(Phc, idx);
//             Score++;
//         } else {
//             BITMAP_CLR(Phc, idx);
//         }
//         if (Mh & Last) {
//             BITMAP_SET(Mhc, idx);
//             Score--;
//         } else {
//             BITMAP_CLR(Mhc, idx);
//         }

//         Ph = (Ph << 1) | Pb;
//         Mh = (Mh << 1) | Mb;

//         Pv = Mh | ~ (Xv | Ph);
//         Mv = Ph & Xv;
//     }
//     return Score;
// }

// static uint64_t myers1999(char *sb1, char *sb2)
// {
//     uint64_t sb1len = strlen(sb1);
//     uint64_t sb2len = strlen(sb2);
//     uint64_t i;
//     uint64_t vmax, hmax;
//     uint64_t *Phc, *Mhc;
//     uint64_t res;

//     if (sb2len == 0)
//         return sb1len;

//     if (sb2len <= 64)
//         return myers1999_simple(sb1, sb2);

//     hmax = CEILDIV(sb1len, 64);
//     vmax = CEILDIV(sb2len, 64);

//     Phc = malloc(hmax * sizeof(uint64_t));
//     Mhc = malloc(hmax * sizeof(uint64_t));


//     for (i = 0; i < hmax; i++) {
//         Mhc[i] = 0;
//         Phc[i] = ~ (uint64_t) 0;
//     }

//     for (i = 0; i < vmax; i++)
//         res = myers1999_block(sb1, sb2, i, Phc, Mhc);

//     free(Phc);
//     free(Mhc);

//     return res;
// }

#define MIN3(a, b, c) ((a) < (b) ? ((a) < (c) ? (a) : (c)) : ((b) < (c) ? (b) : (c)))

int levenshtein(char *s1, char *s2) {
    unsigned int x, y, s1len, s2len;
    s1len = strlen(s1);
    s2len = strlen(s2);
    unsigned int matrix[s2len+1][s1len+1];
    matrix[0][0] = 0;
    for (x = 1; x <= s2len; x++)
        matrix[x][0] = matrix[x-1][0] + 1;
    for (y = 1; y <= s1len; y++)
        matrix[0][y] = matrix[0][y-1] + 1;
    for (x = 1; x <= s2len; x++)
        for (y = 1; y <= s1len; y++)
            matrix[x][y] = MIN3(matrix[x-1][y] + 1, matrix[x][y-1] + 1, matrix[x-1][y-1] + (s1[y-1] == s2[x-1] ? 0 : 1));

    return(matrix[s2len][s1len]);
}


static long get_nanos(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (long)ts.tv_sec * 1000000000L + ts.tv_nsec;
}




#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* alphabet: [a-z0-9] */
const char alphabet[] = "abcdefghijklmnopqrstuvwxyz0123456789";

/**
 * not a cryptographically secure number
 * return interger [0, n).
 */
int intN(int n) { return rand() % n; }

/**
 * Input: length of the random string [a-z0-9] to be generated
 */
char *randomString(int len) {
  char *rstr = malloc((len + 1) * sizeof(char));
  int i;
  for (i = 0; i < len; i++) {
    rstr[i] = alphabet[intN(strlen(alphabet))];
  }
  rstr[len] = '\0';
  return rstr;
}

struct node  { 
  char data; 
  bool endOfWord;
  struct node *next; 
  struct node *sibling; 
}; 

struct node* newNode(int data) { 
  struct node* node = (struct node*)malloc(sizeof(struct node)); 
  node->data = data;  
  node->endOfWord = false;
  node->next = NULL; 
  node->sibling = NULL; 
  return(node); 
}

void insert(struct node *root, char *word) {
  struct node *current = root;

  int i = 0;

  while(1) {
    if(current->data == word[i]) {
      ++i;
      if (current->next == NULL) {
        while(i < strlen(word)) {
          current = current->next = newNode(word[i++]);
        }
        break;
      } 
      else current = current->next;
    } 
    else {
      if (current->sibling == NULL) {
        current = current->sibling = newNode(word[i++]);
        while(i < strlen(word)) {
          current = current->next = newNode(word[i++]);
        }
        break;
      }
      else current = current->sibling;
    }
  }
  current->endOfWord = true;
}  




void padding ( char ch, int n )
{
  int i;
  for ( i = 0; i < n; i++ )
    putchar ( ch );
}
void structure ( struct node *root, int level )
{
  int i;
  if ( root == NULL ) {
    padding ( '\t', level );
    puts ( "~" );
  }
  else {
    structure ( root->sibling, level + 1 );
    padding ( '\t', level );
    printf ( "%c\n", root->data );
    structure ( root->next, level + 1 );
  }
}


int main()
{

  myers1999_simple("asda", "c");
  return 0;
  struct node *root = newNode('a');
  // root->next = newNode('b');
  // root->next->sibling = newNode('j');
  // root->next->next = newNode('c');
  // root->next->next->sibling = newNode('l');
  // root->next->next->sibling->sibling = newNode('m');
  // root->sibling =  newNode('f');
  // root->sibling->next = newNode('g');
  // root->sibling->next->sibling = newNode('s');

  insert(root, "abc");
  insert(root, "abl");
  insert(root, "abm");
  insert(root, "aj");
  insert(root, "fg");
  insert(root, "fs");

  structure(root, 0);

  // printf("%c\n", root->next->data);
  // printf("%c\n", root->next->sibling->data);
  // printf("%c\n", root->next->next->data);
  // printf("%c\n", root->next->next->sibling->data);
  // printf("%c\n", root->next->next->sibling->sibling->data);
  // printf("%c\n", root->sibling->data);
  // printf("%c\n", root->sibling->next->data);
  // printf("%c\n", root->sibling->next->sibling->data);



  // root->next = newNode('b');
  // root->next->next = newNode('c');
  // root->next->sibling = newNode('e');
  // root->next->sibling->sibling = newNode('s');

  // insert(root, "abc");
  // insert(root, "ae");
  // insert(root, "as");
  // insert(root, "asg");
  // insert(root, "aeh");
  // insert(root, "aeha");
  // printf("%c\n", root->next->sibling->next->next->data);
  
  // printf("%c\n", root->next->sibling->next->next->next->data);
  // printf("%c\n", root->next->sibling->sibling->next->data);




    srand(time(NULL));
    long seconds;
    long sec;
    // int i = 0;
    // int o = 0;
    // seconds = get_nanos();
    // while (i < 1000000) {
    //   o += myers1999("sdrwekqlDFlJN", "nisajwkbhfwjdlkchfsjb");
    //   i++;
    // }

    char* array[] = {"nunc", "leo", "at", "sed", "magna", "arcu", "lectus", "Nulla", "dictumst", "purus", "augue", "odio", "Sed", "laoreet", "condimentum", "eleifend", "mattis", "ullamcorper", "massa", "ultrices", "urna", "Proin", "sodales", "leo", "massa", "interdum", "pulvinar", "volutpat", "ut", "nulla", "In", "Quisque", "accumsan", "adipiscing", "feugiat", "at", "lorem", "eget", "congue", "et", "dolor", "amet", "ornare", "ullamcorper", "ullamcorper", "varius", "sapien", "Donec", "dapibus", "amet", "ac", "interdum", "vitae", "ullamcorper", "sed", "justo", "Vestibulum", "Mauris", "faucibus", "Donec", "quis", "eget", "in", "ultrices", "Integer", "mattis", "quis", "Lorem", "tincidunt", "lobortis", "porta", "Maecenas", "sodales", "ante", "Aenean", "gravida", "tristique", "Vivamus", "lacus", "Praesent", "in", "in", "dignissim", "commodo", "ac", "facilisis", "eleifend", "tortor", "elementum", "suscipit", "ornare", "placerat", "turpis", "nibh", "viverra", "ullamcorper", "nunc", "mollis", "Suspendisse", "volutpat", "condimentum", "scelerisque", "ipsum", "amet", "hendrerit", "posuere", "vitae", "dapibus", "vestibulum", "efficitur", "Etiam", "mi", "tellus", "nisi", "ex", "Maecenas", "ultricies", "lacus", "Proin", "nascetur", "Vestibulum", "sapien", "magna", "vel", "vestibulum", "erat", "dolor", "fermentum", "leo", "nunc", "urna", "nunc", "arcu", "id", "massa", "imperdiet", "vitae", "facilisis", "orci", "ut", "massa", "eget", "quis", "tincidunt", "tortor", "Suspendisse", "quis", "porta", "Nulla", "sit", "interdum", "pharetra", "viverra", "pretium", "ligula", "turpis", "semper", "blandit", "nisi", "penatibus", "rutrum", "lobortis", "at", "faucibus", "porta", "scelerisque", "scelerisque", "libero", "turpis", "quis", "amet", "mi", "ultrices", "vitae", "Nam", "egestas", "nec", "lacus", "maximus", "amet", "at", "diam", "amet", "sed", "blandit", "pretium", "congue", "Vestibulum", "posuere", "blandit", "risus", "porta", "Morbi", "libero", "lectus", "aliquam", "ultrices", "potenti", "aliquet", "nunc", "neque", "tincidunt", "Integer", "et", "leo", "ultrices", "ligula", "velit", "efficitur", "tristique", "urna", "nisi", "metus", "ornare", "Fusce", "a", "sem", "elit", "sem", "est", "nisl", "molestie", "posuere", "laoreet", "nec", "lacinia", "lacus", "sagittis", "odio", "Nam", "tincidunt", "arcu", "ultrices", "laoreet", "augue", "Maecenas", "non", "libero", "Nulla", "nisl", "quam", "in", "malesuada", "ligula", "quam", "aliquam", "Praesent", "non", "vel", "posuere", "posuere", "ridiculus", "ante", "Vestibulum", "dis", "nisl", "feugiat", "Vivamus", "elementum", "ornare", "sagittis", "posuere", "porttitor", "auctor", "venenatis", "mauris", "ipsum", "eget", "eget", "primis", "sed", "ullamcorper", "pretium", "cursus", "vitae", "non", "lobortis", "ipsum", "nisi", "malesuada", "ut", "urna", "Nulla", "Suspendisse", "accumsan", "est", "velit", "porttitor", "quis", "dolor", "suscipit", "nibh", "vitae", "mattis", "magna", "luctus", "accumsan", "faucibus", "Nulla", "Suspendisse", "vel", "ac", "porta", "non", "dignissim", "laoreet", "mattis", "placerat", "placerat", "Nunc", "tincidunt", "ultricies", "Nam", "et", "Aliquam", "gravida", "congue", "dui", "semper", "sit", "lectus", "ac", "ante", "rhoncus", "vel", "lobortis", "est", "Praesent", "amet", "sem", "feugiat", "diam", "elementum", "laoreet", "sit", "augue", "rutrum", "non", "magna", "accumsan", "dolor", "ligula", "laoreet", "lorem", "condimentum", "felis", "Praesent", "lacus", "nisi", "lacus", "eu", "eget", "amet", "vitae", "vitae", "turpis", "varius", "aliquam", "amet", "ut", "at", "magna", "risus", "ut", "eu", "ultricies", "Morbi", "amet", "faucibus", "mauris", "ultricies", "lacus", "tincidunt", "diam", "ac", "Duis", "tempor", "dui", "ut", "Maecenas", "mi", "nec", "Orci", "magnis", "feugiat", "auctor", "risus", "vitae", "mi", "urna", "nunc", "In", "auctor", "sodales", "et", "Aenean", "vel", "elit", "Nam", "vel", "eu", "et", "vitae", "euismod", "lacus", "Fusce", "mattis", "Curabitur", "tempus", "suscipit", "odio", "rutrum", "ac", "vitae", "nec", "malesuada", "vel", "blandit", "ante", "Donec", "dapibus", "elit", "Morbi", "aliquam", "magna", "facilisis", "diam", "orci", "erat", "tortor", "nulla", "tempus", "et", "at", "mattis", "sit", "nec", "purus", "aliquet", "dapibus", "laoreet", "Sed", "justo", "semper", "ac", "eu", "ut", "leo", "et", "sem", "tincidunt", "egestas", "Nulla", "eget", "consectetur", "consectetur", "consectetur", "massa", "quam", "dapibus", "tempus", "In", "aliquam", "ac", "ac", "aliquet", "nec", "gravida", "quis", "rutrum", "erat", "efficitur", "feugiat", "Aliquam", "porta", "mattis", "odio", "erat", "vel", "ut", "In", "Vestibulum", "nunc", "odio", "sodales", "a", "metus", "amet", "at", "tristique", "eget", "nec", "convallis", "quis", "lacus", "suscipit", "velit", "nisi", "odio", "odio", "urna", "nisl", "eget", "nec", "vestibulum", "hendrerit", "nibh", "erat", "mus", "felis", "eu", "fringilla", "molestie", "a", "sed", "amet", "felis", "amet", "mauris", "sit", "justo", "nulla", "lorem", "In", "elit", "vel", "velit", "diam", "eu", "lacinia", "Nulla", "in", "massa", "ex", "quam", "varius", "et", "adipiscing", "Aliquam", "non", "rhoncus", "porta", "eget", "Suspendisse", "hendrerit", "venenatis", "nulla", "cursus", "fames", "lorem", "ornare", "nulla", "nunc", "cursus", "interdum", "lacus", "pulvinar", "eu", "malesuada", "elementum", "porttitor", "condimentum", "aliquet", "Fusce", "pulvinar", "Sed", "quam", "ut", "ornare", "sem", "sodales", "tempor", "vulputate", "montes", "non", "pulvinar", "fermentum", "orci", "quis", "eget", "at", "enim", "purus", "justo", "lorem", "et", "urna", "ut", "et", "sit", "vel", "nec", "porta", "eu", "ut", "Phasellus", "ullamcorper", "ut", "ut", "auctor", "arcu", "Lorem", "id", "tempor", "enim", "luctus", "Aliquam", "volutpat", "Donec", "ultrices", "sodales", "sit", "eget", "Vestibulum", "Integer", "luctus", "sem", "interdum", "quis", "Aenean", "turpis", "In", "nisl", "lectus", "quam", "lobortis", "Integer", "massa", "lobortis", "tortor", "Quisque", "mollis", "fringilla", "sit", "natoque", "volutpat", "eu", "congue", "velit", "consectetur", "aliquet", "sit", "pulvinar", "Sed", "ultrices", "vulputate", "ut", "ornare", "Quisque", "Duis", "eu", "Nulla", "tempus", "elementum", "ullamcorper", "ultricies", "auctor", "cursus", "nec", "iaculis", "ipsum", "magna", "lobortis", "nec", "pulvinar", "Integer", "volutpat", "dictum", "luctus", "malesuada", "metus", "Duis", "justo", "nibh", "metus", "quis", "viverra", "Etiam", "faucibus", "velit", "eu", "augue", "semper", "sapien", "ut", "eu", "eu", "Nam", "justo", "sapien", "Ut", "at", "tempus", "mi", "tempus", "leo", "Ut", "arcu", "a", "consequat", "lorem", "mauris", "eros", "ligula", "non", "mi", "ligula", "odio", "vel", "suscipit", "augue", "mi", "viverra", "Sed", "vitae", "mollis", "lacinia", "Cras", "consequat", "eu", "urna", "nunc", "metus", "in", "elit", "ut", "blandit", "aliquet", "justo", "Donec", "a", "ligula", "lorem", "non", "faucibus", "finibus", "erat", "volutpat", "ligula", "leo", "sit", "fermentum", "laoreet", "in", "faucibus", "ante", "luctus", "libero", "Mauris", "est", "arcu", "maximus", "nec", "tincidunt", "Donec", "purus", "sed", "Etiam", "lacinia", "tristique", "malesuada", "dolor", "hac", "nec", "interdum", "nec", "iaculis", "pharetra", "Vivamus", "vel", "id", "vitae", "lectus", "viverra", "Nam", "mauris", "aliquet", "hendrerit", "lorem", "iaculis", "Etiam", "Mauris", "est", "Aliquam", "Sed", "egestas", "metus", "sodales", "nisl", "Cras", "purus", "nulla", "pharetra", "a", "auctor", "Etiam", "Aenean", "urna", "id", "In", "porttitor", "vel", "leo", "Nullam", "posuere", "in", "sapien", "a", "a", "purus", "nulla", "dui", "feugiat", "velit", "dapibus", "sem", "justo", "Fusce", "hendrerit", "lacus", "non", "eu", "elit", "lectus", "commodo", "nec", "Mauris", "vehicula", "augue", "cursus", "ultrices", "auctor", "consectetur", "eu", "potenti", "dictum", "non", "nulla", "Vivamus", "vulputate", "viverra", "malesuada", "venenatis", "metus", "pharetra", "varius", "accumsan", "nunc", "posuere", "suscipit", "lacus", "Nunc", "nisi", "feugiat", "volutpat", "tempor", "metus", "massa", "nulla", "elit", "lorem", "euismod", "platea", "nunc", "pharetra", "mauris", "Sed", "facilisis", "In", "ornare", "iaculis", "Suspendisse", "et", "est", "maximus", "eu", "sed", "Suspendisse", "fermentum", "id", "pretium", "lectus", "et", "nulla", "sem", "malesuada", "ornare", "habitasse", "Duis", "felis", "leo", "massa", "eget", "ligula", "orci", "Nunc", "viverra", "vitae", "Donec", "leo", "pharetra", "sodales", "eu", "dictum", "ex", "non", "Sed", "elit", "fringilla", "scelerisque", "feugiat", "semper", "efficitur", "vehicula", "felis", "tempor", "leo", "dui", "rutrum", "nisl", "tortor", "mollis", "pretium", "ante", "consectetur", "parturient", "Aenean", "fringilla", "consequat", "eu", "condimentum", "Aenean", "malesuada", "dui", "orci", "id", "ut", "mattis", "leo", "Praesent", "tortor", "feugiat", "placerat", "Vestibulum", "pulvinar", "eu", "pharetra", "a", "felis", "sit", "faucibus", "gravida", "id", "scelerisque", "bibendum", "leo", "sed", "pulvinar", "convallis", "elit", "augue", "justo", "ornare", "Donec", "scelerisque", "sed", "Maecenas", "ultrices", "hendrerit", "ut", "nec", "porttitor", "sit", "Morbi", "tempus", "sit", "vel", "mollis", "enim", "ut", "dignissim", "elementum", "tempor", "eleifend", "et", "Interdum", "eu", "Vivamus", "ex", "fermentum", "massa", "erat", "Praesent", "elementum", "molestie", "feugiat", "quis", "mauris", "turpis", "lorem"};

    int wl = 1000;
    int o = 0;
    int i = 0;
    int j = 0;

    
    // while (j < 1000000000) {
    //   while (i < wl) {
    //   o += myers1999_simple(array[i], array[i + 1]);
    //   i += 2;
    //   }
    //   j++;
    // }
    // sec = get_nanos();
    // printf("%ld\n", sec - seconds); 

    char *p;
    char *q;
    int k = 0;
    unsigned int miss = 0;
    while (k < 100000) {
      p = randomString(64);
      q = randomString(64);
      if (myers1999_simple(q, p) != levenshtein(q, p)) {
        miss++;
      }
      k++;
    } 
    printf("%d\n", miss);


    

    seconds = get_nanos();
    k = 0;
    miss = 0;
    while (k < 400000) {
      p = randomString(64);
      q = randomString(64);
      miss += myers1999_simple(p, q);
      k++;
    } 

    sec = get_nanos();
    printf("%ld\n", sec - seconds); 

    


    // printf("%d\n", o);
    // printf("%d\n", (int)myers1999_simple("½+Ǵ=gffgdfy", "sgfyfhwiuqofgjshkjf"));
    // printf("%lu\n", myers1999("sdrwekqbnmfyrfgfrlDFlJNwfwfwf", "nisaJMSMADKAW8765trfdvgty54refdrtrejsmfg43lkwrjdvbjenkwqefterfDKADWKADKjwkbhfwjtreerfjdlkchfsjb")); 
    // printf("%ld\n", sec - seconds); 
    // printf("%d\n", o);
    return 0;
}