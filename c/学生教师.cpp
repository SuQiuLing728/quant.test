#include<iostream>
using namespace std;

class person{
    public:
    string name;
    int id,gender,age;
    public:
    void setage(int a) {age = a ;}
    void coutage() {cout<<age<<endl;}
    void setid(int a) {id = a ;}
    void coutid() {cout<<id<<endl;}
    void setname(string a) {name = a ;}
    void coutname() {cout<<name<<endl;}
    void setgender(int a) {gender = a ;}
    void coutgender() {cout<<gender<<endl;}

};

class Student:public person{
    public:
    int grades;
    int goals;

};

class Teacher:public person{
    public:
    int department;
    int job_title;
    
};