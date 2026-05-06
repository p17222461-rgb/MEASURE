from pydantic import BaseModel, Field
from typing import List

class PersonalInfo(BaseModel):
    full_name: str = Field(default="", description="Full name of the person")
    email: str = Field(default="", description="Email address")
    phone: str = Field(default="", description="Phone number")
    location: str = Field(default="", description="Location/address")
    github: str = Field(default="", description="GitHub profile URL")
    linkedin: str = Field(default="", description="LinkedIn profile URL")
    address: str = Field(default="", description="Physical address")
    identification_number: str = Field(default="", description="Identification number")

class WorkExperience(BaseModel):
    company: str = Field(default="", description="Company name")
    role: str = Field(default="", description="Job role/title")
    start_date: str = Field(default="", description="Start date")
    end_date: str = Field(default="", description="End date")
    employment_description: str = Field(default="", description="Description of employment")
    location: str = Field(default="", description="Location of the company")

class Certification(BaseModel):
    name: str = Field(default="", description="Certification name")
    institution: str = Field(default="", description="Issuing organization (if any)")
    date_obtained: str = Field(default="", description="Date or year obtained")
    license: str = Field(default="", description="License number (if any)")

class EducationDegree(BaseModel):
    institution: str = Field(default="", description="Educational institution")
    degree: str = Field(default="", description="Degree obtained")
    start_date: str = Field(default="", description="Start date")
    end_date: str = Field(default="", description="End date")
    location: str = Field(default="", description="Location of institution")

class Education(BaseModel):
    degrees: List[EducationDegree] = Field(default_factory=list, description="Formal degree-based education")
    certifications: List[Certification] = Field(default_factory=list, description="Professional or academic certifications not tied to a degree")

class ResumeData(BaseModel):
    personal_info: PersonalInfo = Field(default_factory=PersonalInfo, description="Personal information")
    skills: List[str] = Field(default_factory=list, description="List of skills including known languages")
    work_experience: List[WorkExperience] = Field(default_factory=list, description="Work experience")
    education: Education = Field(default_factory=Education, description="Education history including degrees and certifications")