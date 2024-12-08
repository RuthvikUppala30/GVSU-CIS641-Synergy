# Software Requirements Specification (SRS)

## Overview
This document serves as the Software Requirements Specification (SRS) for the University Chatbot System project. The chatbot is designed to assist students, staff, and faculty with streamlined access to university-related services, providing an intuitive conversational interface. This document outlines the functional and non-functional requirements, traceability to artifacts, and the change management plan, ensuring clarity, testability, and alignment with the project's goals.

---

## Software Requirements

### Functional Requirements
Below are the system's functional requirements, categorized into key features.

#### User Authentication and Account Management
| ID   | Requirement |
|------|-------------|
| FR1  | The system shall allow users to create accounts using their university email. |
| FR2  | The chatbot shall support secure login via username and password. |
| FR3  | Users shall be able to reset their password via an email verification process. |
| FR4  | The chatbot shall restrict access to certain features based on user roles (e.g., student, staff). |
| FR5  | The system shall allow users to update their profile information. |

#### Information Retrieval
| ID   | Requirement |
|------|-------------|
| FR6  | The chatbot shall provide information about university policies. |
| FR7  | Users shall be able to query the chatbot for building locations and directions. |
| FR8  | The system shall retrieve and display contact information for faculty and staff. |
| FR9  | The chatbot shall provide real-time weather updates for the university's location. |
| FR10 | The system shall support keyword-based search functionality for FAQs. |

#### Course and Academic Management
| ID   | Requirement |
|------|-------------|
| FR11 | The chatbot shall allow users to check their course schedules. |
| FR12 | The system shall provide access to course descriptions and prerequisites. |
| FR13 | Users shall be able to inquire about their grades and academic progress. |
| FR14 | The chatbot shall integrate with the university’s LMS to display assignments. |
| FR15 | Users shall be able to receive reminders for upcoming exams and deadlines. |

#### Event and Schedule Management
| ID   | Requirement |
|------|-------------|
| FR16 | The chatbot shall provide a calendar of university events. |
| FR17 | Users shall be able to set reminders for selected events. |
| FR18 | The system shall notify users of event cancellations or changes. |
| FR19 | The chatbot shall suggest events based on user preferences. |
| FR20 | Users shall be able to book rooms for study sessions or meetings via the chatbot. |

#### Feedback and Support
| ID   | Requirement |
|------|-------------|
| FR21 | The chatbot shall allow users to submit feedback about its functionality. |
| FR22 | The system shall provide troubleshooting guides for common issues. |
| FR23 | Users shall be able to report bugs or errors directly through the chatbot. |
| FR24 | The chatbot shall escalate unresolved issues to human support staff. |
| FR25 | The system shall track the resolution status of reported issues. |

---

### Non-Functional Requirements
Below are the system's quality and performance requirements.

#### Performance
| ID   | Requirement |
|------|-------------|
| NFR1  | The chatbot shall respond to user queries within 2 seconds under normal load conditions. |
| NFR2  | The system shall support at least 100 concurrent users without performance degradation. |
| NFR3  | The chatbot shall maintain an average uptime of 99.9% per month. |
| NFR4  | The system shall process data requests and deliver results with an accuracy rate of 95%. |
| NFR5  | The chatbot shall handle database queries efficiently, with no more than 5% of queries exceeding 1 second. |

#### Scalability
| ID   | Requirement |
|------|-------------|
| NFR6  | The chatbot shall scale horizontally to accommodate an increase in user traffic. |
| NFR7  | The system shall support integration with additional university systems as needed. |
| NFR8  | The chatbot shall allow adding new features without significant architectural changes. |
| NFR9  | The system shall accommodate an increase in data storage by up to 50% annually. |
| NFR10 | The chatbot shall scale to support up to 500 concurrent users during peak periods. |

#### Security
| ID   | Requirement |
|------|-------------|
| NFR11 | The chatbot shall encrypt all user data in transit using SSL/TLS protocols. |
| NFR12 | The system shall ensure that only authorized users can access restricted features. |
| NFR13 | The chatbot shall comply with university data privacy policies. |
| NFR14 | The system shall log all access and modification attempts for auditing purposes. |
| NFR15 | The chatbot shall automatically lock user accounts after three consecutive failed login attempts. |

#### Usability
| ID   | Requirement |
|------|-------------|
| NFR16 | The chatbot shall provide a user-friendly interface for both desktop and mobile platforms. |
| NFR17 | The system shall offer contextual help for all major functionalities. |
| NFR18 | The chatbot shall use natural language processing to understand user queries effectively. |
| NFR19 | The system shall support both text-based and voice-based interactions. |
| NFR20 | The chatbot shall provide error messages that clearly explain how users can correct their input. |

#### Reliability and Availability
| ID   | Requirement |
|------|-------------|
| NFR21 | The chatbot shall recover from system crashes within 5 minutes. |
| NFR22 | The system shall ensure consistent functionality even under high traffic conditions. |
| NFR23 | The chatbot shall undergo monthly maintenance to address software updates. |
| NFR24 | The system shall have a fallback mechanism to redirect users to a human operator during downtime. |
| NFR25 | The chatbot shall monitor its own performance and report anomalies in real-time. |

---

## Change Management Plan

### Training
- Interactive tutorials embedded within the chatbot for first-time users.
- Comprehensive user manuals covering all features.
- Virtual training sessions for staff to familiarize them with administrative functions.

### Integration
- APIs to integrate the chatbot with SIS, LMS, and event management tools.
- Testing in a staging environment to ensure compatibility with existing systems.

### Issue Resolution
- Bug reporting system within the chatbot interface.
- Regular software updates to fix issues and add features.
- Escalation to live support for unresolved problems.

---

## Traceability Links

### Use Case Diagram Traceability
| Artifact ID | Artifact Name                  | Requirement ID                      |
|-------------|--------------------------------|-------------------------------------|
| UC1         | Check Admission Status         | FR6, FR7                            |
| UC2         | Ask FAQs                       | FR10                                |
| UC3         | Book Campus Visits             | FR19, FR20                          |
| UC4         | Contact Admissions and Services| FR21, FR22                          |
| UC5         | Check Classes                  | FR11, FR12                          |
| UC6         | Update Availability Status     | NFR2, NFR9                          |
| UC7         | Check HR Details               | FR8                                 |
| UC8         | Book Meeting Rooms             | FR19, FR20                          |
| UC9         | Monitor Performance            | NFR3, NFR25                         |
| UC10        | Resolve Queries                | FR23, FR24                          |
| UC11        | View System Report             | NFR21, NFR22                        |
| UC12        | Manage Content                 | NFR7, NFR10                         |

### Class Diagram Traceability
| Artifact Name         | Requirement ID                     |
|-----------------------|------------------------------------|
| Authentication Class  | FR1, FR2, FR3                     |
| Information Class     | FR6, FR7, FR8                     |
| Booking Class         | FR19, FR20                        |
| Feedback Class        | FR21, FR23, FR24                  |
| Performance Class     | NFR3, NFR25                       |
| Admin Management Class| FR23, NFR10, NFR21                |

### Activity Diagram Traceability
| Artifact ID  | Artifact Name                | Requirement ID                     |
|--------------|------------------------------|------------------------------------|
| AD1          | Login Activity Diagram       | FR1, FR2, FR3                     |
| AD2          | Information Query Workflow   | FR6, FR7, FR8, FR10               |
| AD3          | Room Booking Workflow        | FR19, FR20                        |
| AD4          | Feedback Processing Workflow | FR21, FR23, FR24                  |
| AD5          | Admin Report Workflow        | NFR21, NFR22                      |

## Project Artifacts
- [Midterm SRS](https://github.com/RuthvikUppala30/GVSU-CIS641-Synergy/blob/main/docs/Requirement_Specification_Document.md)
- [Diagrams Folder](https://github.com/your-repo-link/diagrams)
- [Final SRS](https://github.com/your-repo-link/software_requirements_specification_final.md)
