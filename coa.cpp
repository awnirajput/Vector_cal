#include<bits/stdc++.h>
using namespace std;

string clean_string(string);
string reverse_string(string);
bool isValid(string);
void set_operator();
bool isChar(char);

// to store operator sign with corresponding operation
map<char, string> opr;

int main()
{
    int sz;
    char ch;
    string prefix, str;
    stack<string> operand_stk;

    cout << "Enter prefix: ";
    getline(cin, prefix);
    cout << endl;


    // remove the white spaces from the string
    prefix = clean_string(prefix);
    str = prefix;


    // reverse the string
    prefix = reverse_string(prefix);


    // get the length of prefix string
    sz = (int) prefix.size();


    // convert prefix to uppercase form
    for(int i = 0; i < sz; i++)
    {
        ch = prefix[i];
        ch = toupper(ch);
        prefix[i] = ch;
    }


    // check if the input is valid
    if(!isValid(prefix))
    {
        cout << "Input is INVALID !!" << endl;
        return 0;
    }
    else
    {
        cout << "Successfully generated assembly code !!" << endl;
        cout << "Assembly code is saved into 'assemblyCode.txt' file" << endl;
    }


    // set operator list
    set_operator();


    // show the output in a text file
    // this output is the assembly code
    freopen("assemblyCode.txt", "w", stdout);


    /// assembly part starts
    cout << "Input prefix: " << str << endl;
    cout << "Reversed prefix: " << prefix << endl << endl;
    cout << "Assembly Code: " << endl;
    /// assembly part ends


    char operator1;
    string s, result;
    string operation;
    string operand1, operand2;
    string new_operand;

    for(int i = 0; i < sz; i++)
    {
        ch = prefix[i];

        if(isChar(ch))
        {
            // if character is an operand, push it in operand stack
            s = ch;
            operand_stk.push(s);


            /// assembly part starts
            cout << "PUSH " << ch << endl;
            /// assembly part ends
        }

        else
        {
            // storing operand1
            operand1 = operand_stk.top();
            // pop out operand1 from stack
            operand_stk.pop();


            // storing operand2
            operand2 = operand_stk.top();
            // pop out operand2 from stack
            operand_stk.pop();


            // storing operator1
            operator1 = ch;


            // from operand1 and operand2, make new operand
            new_operand = '(' + operand1 + operator1 + operand2 + ')';
            // push the new operand into operand stack
            operand_stk.push(new_operand);


            /// assembly part starts
            cout << "POP R1             ; R1 = " << operand1 << endl;
            cout << "POP R2             ; R2 = " << operand2 << endl;


            // fetch the corresponding operation of the operator from operator list
            operation = opr[operator1];
            cout << operation;


            cout << "R1, R1, R2     ; R1 = R1 " << operator1 << " R2" << endl;
            cout << "PUSH R1" << endl;
            /// assembly part ends
        }
    }


    // store the final result
    result = operand_stk.top();
    operand_stk.pop();


    /// assembly start
    cout << "POP Result         ; Result = " << result << endl;
    /// assembly end

    return 0;
}



string clean_string(string str)
{
    int sz;
    char ch;
    string s;

    sz = (int) str.size();
    for(int i = 0; i < sz; i++)
    {
        ch = str[i];
        if(ch != ' ' && ch != '(' && ch != ')')
        {
            s.push_back(ch);
        }
    }

    return s;
}

string reverse_string(string str)
{
    int sz;
    char ch;
    string s;

    sz = (int) str.size();
    for(int i = sz-1; i >= 0; i--)
    {
        ch = str[i];
        s.push_back(ch);
    }

    return s;
}

bool isValid(string str)
{
    char ch;
    int sz;
    int count_operand = 0;

    sz = (int) str.size();
    for(int i = 0; i < sz; i++)
    {
        ch = str[i];
        if('A' <= ch && ch <= 'Z')
        {
            count_operand++;
        }
        else
        {
            count_operand--;
            if(count_operand <= 0)
            {
                return false;
            }
        }
    }

    if(count_operand == 1)
    {
        return true;
    }
    else
    {
        return false;
    }
}

void set_operator()
{
    opr['+'] = "ADD ";
    opr['-'] = "SUB ";
    opr['*'] = "MPY ";
    opr['/'] = "DIV ";
    opr['%'] = "MOD ";
    opr['^'] = "POW ";

    return;
}

bool isChar(char ch)
{
    if('A' <= ch && ch <= 'Z')
    {
        return true;
    }
    else
    {
        return false;
    }
}

//      SAMPLE INPUTS:

//      (+(*AB)(/CD))
//      /*P+QRS
//      *   W   +   X  / Y   Z