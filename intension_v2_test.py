from llm import LLM  # Import the LLM class
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import RegexParser

class Intension(LLM):  # Inherit from LLM
    """Represents a zero-shot chain-of-thought implementing an intension for triples."""

    PROMPT_TEMPLATE = """
Determine the truth value of following knowledge graph triple 
in a hypothetical world where the following is true:
{graph}

Let's think step by step. Provide a rationale for 
your decision, then based on that rationale,
provide an answer of 1 if true, otherwise 
provide an answer of 0.
###
Subject: <vrd:skateOn>
Predicate: <rdfs:domain>
Object: <vrd:PullCapableThing>
Rationale: The given knowledge graph triple is true based on the ontology information provided.

The ontology states that the property 'vrd:skateOn' has the following definition:

vrd:skateOn rdf:type owl:ObjectProperty ;
            rdfs:subPropertyOf vrd:on ;
            rdfs:domain vrd:Person ;
            rdfs:range vrd:SkateboardingRelatedThing ;
            rdfs:comment "Property 'skateOn' has a consistent meaning/usage relating to skateboarding."@en .

This indicates that the domain of the 'vrd:skateOn' property is the class 'vrd:Person'. The ontology also defines the class 'vrd:PullCapableThing' as:

vrd:PullCapableThing rdf:type owl:Class ;
                     owl:equivalentClass [ rdf:type owl:Class ;
                                           owl:unionOf ( vrd:Horse
                                                         vrd:Person
                                                         vrd:TrainEngine
                                                         vrd:Truck
                                                       )
                                         ] ;
                     rdfs:subClassOf vrd:MixedEnvironmentThing ;
                     rdfs:comment "Something that is capable of pulling something. Something that can be said to 'pull' something else. Defined to serve as a domain restriction for object property 'pull'."@en .

Since 'vrd:Person' is a subclass of 'vrd:PullCapableThing', the triple stating that the domain of 'vrd:skateOn' is 'vrd:PullCapableThing' is true according to the ontology.
Answer: 1
###
Subject: <vrd:sitOn>
Predicate: <rdfs:domain>
Object: <owl:Thing>
Rationale: The given knowledge graph triple is true in the context of the provided ontology.

The ontology states that the property `<vrd:sitOn>` has the domain `<owl:Thing>`, which means that the `<vrd:sitOn>` property can be applied to any individual, regardless of its class. This is consistent with the information provided in the knowledge graph triple.

While the ontology also defines `<vrd:sitOn>` as a subproperty of `<vrd:on>` with a domain restriction to `vrd:Mammal`, this does not contradict the statement in the knowledge graph triple. The domain of `<vrd:sitOn>` being `<owl:Thing>` is a more general statement that encompasses the more specific domain restriction to `vrd:Mammal`.

Therefore, the knowledge graph triple `<vrd:sitOn> <rdfs:domain> <owl:Thing>` is true in the context of the provided ontology.
Answer: 1
###
Subject: <xsd:float>
Predicate: <owl:sameAs>
Object: <xsd:float>
Rationale: The given knowledge graph triple stating that the subject is an xsd:float, the predicate is owl:sameAs, and the object is an xsd:float is true.

The ontology provided does not contain any information about XML Schema datatypes or OWL properties, as it is focused on modeling the domain of the VRD dataset and its visual relationship annotations. However, the owl:sameAs predicate is a standard OWL property that can be used to state that two individuals are the same. 

In the context of the hypothetical world described, the triple could be interpreted as stating that two floating-point numbers (represented using the xsd:float datatype) are the same or equivalent. This is a valid use of the owl:sameAs predicate, as it allows expressing the equivalence of two individuals, regardless of whether they represent objects in the VRD dataset or other types of entities.

Therefore, the truth value of the given knowledge graph triple is true, as it represents a valid use of the owl:sameAs predicate to state the equivalence of two xsd:float values, even though the provided ontology does not contain any specific information about these datatypes or properties.
Answer: 1
###
Subject: <owl:Nothing>
Predicate: <rdfs:subClassOf>
Object: <vrd:MotionCapableThing>
Rationale: The given knowledge graph triple is true.

The class <owl:Nothing> is the empty class, which means it has no instances. By definition, the empty class is a subclass of every other class, including <vrd:MotionCapableThing>. This is because the empty class satisfies the necessary and sufficient conditions for being a subclass of any other class - it has no instances, and therefore all instances of the empty class are also instances of any other class.

Therefore, the statement that <owl:Nothing> is a subclass of <vrd:MotionCapableThing> is true.
Answer: 1
###
Subject: <vrd:playWith>
Predicate: <rdfs:domain>
Object: <vrd:NaturalEnvironmentThing>
Rationale: The given knowledge graph triple is true in the context of the provided ontology.

The ontology states that the property <vrd:playWith> has the domain <vrd:NaturalEnvironmentThing>. This means that the subject of the <vrd:playWith> property must be an instance of the class <vrd:NaturalEnvironmentThing> or one of its subclasses.

In the hypothetical world described by the ontology, the class <vrd:NaturalEnvironmentThing> is defined as the top-level class for all naturally occurring things in the VRD material world domain. This class has several subclasses, including <vrd:Mammal>, which in turn has subclasses such as <vrd:Person>.

The property <vrd:playWith> is defined as a subproperty of <vrd:use>, and its domain is restricted to the class <vrd:Person>. This means that instances of the <vrd:Person> class can be the subject of the <vrd:playWith> property, and the object must be an instance of the <vrd:NaturalEnvironmentThing> class or one of its subclasses.

Therefore, the given knowledge graph triple, which states that the property <vrd:playWith> has the domain <vrd:NaturalEnvironmentThing>, is true in the context of the provided ontology.
Answer: 1
###
Subject: <vrd:drive>
Predicate: <rdfs:domain>
Object: <vrd:CarryCapableMammal>
Rationale: The given knowledge graph triple is true in the hypothetical world described by the provided ontology.

The ontology defines the class 'CarryCapableMammal' as the union of the classes 'Elephant', 'Horse', and 'Person'. This means that instances of these three classes are considered to be capable of carrying things.

The ontology also defines the property 'drive' as a subproperty of 'use', with the domain being 'Person' and the range being 'DrivableMotorisedVehicle'. This indicates that the 'drive' property is specifically used to describe a person driving a motorized vehicle.

However, the ontology also states that the 'drive' property has the domain 'CarryCapableMammal'. This means that the 'drive' property can be applied not only to instances of the 'Person' class, but also to instances of the 'Elephant' and 'Horse' classes, as they are all considered to be 'CarryCapableMammal' entities.

Therefore, the knowledge graph triple stating that the 'drive' property has the domain 'CarryCapableMammal' is true in the context of the provided ontology.
Answer: 1
###
Subject: <vrd:EnclosingArchitecturalStructure>
Predicate: <rdfs:subClassOf>
Object: <vrd:EnclosingArchitecturalStructure>
Rationale: The given knowledge graph triple is true. The statement that <vrd:EnclosingArchitecturalStructure> is a subclass of itself is a valid and common ontological construct.

In ontologies, it is perfectly acceptable and often necessary for a class to be a subclass of itself. This is known as a reflexive subclass relationship. It indicates that the class is a subtype of itself, meaning that all instances of the class are also instances of the class.

Reflexive subclass relationships are useful for modeling situations where a class has a hierarchical relationship with itself. For example, in the given ontology, the class <vrd:EnclosingArchitecturalStructure> likely represents a broad category of architectural structures that enclose or contain other things. It is reasonable for this class to be defined as a subclass of itself, as any instance of an enclosing architectural structure is also an instance of the broader class of enclosing architectural structures.

Reflexive subclass relationships do not lead to logical inconsistencies, as long as they are used appropriately within the context of the ontology. They help to capture the inherent self-referential nature of certain classes and their hierarchical relationships.

Therefore, the given knowledge graph triple, <vrd:EnclosingArchitecturalStructure> rdfs:subClassOf <vrd:EnclosingArchitecturalStructure>, is a valid and true statement within the context of the VRD-World ontology.
Answer: 1
###
Subject: <vrd:sleepNextTo>
Predicate: <rdfs:domain>
Object: <vrd:NaturalEnvironmentThing>
Rationale: The given knowledge graph triple is true in the hypothetical world described by the provided ontology.

The ontology defines the property `<vrd:sleepNextTo>` as a subproperty of `<vrd:nextTo>`, and the domain of `<vrd:sleepNextTo>` is defined as `vrd:Mammal`. The class `vrd:Mammal` is a subclass of `vrd:NaturalEnvironmentEarthBoundThing`, which is a subclass of `vrd:NaturalEnvironmentThing`.

Therefore, the domain of the `<vrd:sleepNextTo>` property is a subclass of `<vrd:NaturalEnvironmentThing>`, which means that the knowledge graph triple `<vrd:sleepNextTo> <rdfs:domain> <vrd:NaturalEnvironmentThing>` is true in the given ontology.

The ontology hierarchy ensures that any instance of the `vrd:Mammal` class, which is the domain of `<vrd:sleepNextTo>`, is also an instance of the `<vrd:NaturalEnvironmentThing>` class. This satisfies the domain restriction specified in the given knowledge graph triple.
Answer: 1
###
Subject: <vrd:walk>
Predicate: <rdfs:domain>
Object: <vrd:TalkToableThing>
Rationale: The knowledge graph triple <vrd:walk> <rdfs:domain> <vrd:TalkToableThing> is true in the hypothetical world described by the given ontology.

The ontology defines the class <vrd:TalkToableThing> as the union of <vrd:Person> and <vrd:Phone>. The property <vrd:walk> is defined to have the domain <vrd:Person>, which is a subclass of <vrd:TalkToableThing>. 

According to the OWL semantics, if a property has a domain restriction to a class C, then it is also valid to state that the domain of the property is any superclass of C. In this case, since <vrd:Person> is a subclass of <vrd:TalkToableThing>, the triple <vrd:walk> <rdfs:domain> <vrd:TalkToableThing> is true.

The ontology also states that the property <vrd:walk> "is highly specific and used in (multiple instances of) only 1 distinct visual relationship: (person, walk, dog)." This does not contradict the fact that the domain of <vrd:walk> can be stated as the superclass <vrd:TalkToableThing>, as the ontology allows for more general domain and range restrictions than the specific usage examples.

Therefore, the given knowledge graph triple is true in the hypothetical world described by the ontology.
Answer: 1
###
Subject: <owl:Nothing>
Predicate: <rdfs:subClassOf>
Object: <vrd:Cart>
Rationale: The given knowledge graph triple stating that <owl:Nothing> is a subclass of <vrd:Cart> is true.

The class <owl:Nothing> is a special class in OWL that represents the empty set. By definition, the empty set is a subset of any other set, including the set represented by the class <vrd:Cart>. Therefore, <owl:Nothing> is a subclass of <vrd:Cart>, as per the semantics of the OWL subclass relationship.

This is a valid and true statement in the context of the VRD-World ontology, as <vrd:Cart> is defined as a subclass of <vrd:Vehicle>, which is a subclass of <vrd:EngineeredEnvironmentThing>. The class <owl:Nothing> being a subclass of <vrd:Cart> means that the empty set is a subset of the set of all carts, which is a valid logical relationship.
Answer: 1
###
Subject: <vrd:standNextTo>
Predicate: <rdfs:subPropertyOf>
Object: <vrd:standNextTo>
Rationale: The given knowledge graph triple:

Subject: <vrd:standNextTo>
Predicate: <rdfs:subPropertyOf>
Object: <vrd:standNextTo>

is true in the hypothetical world described by the provided ontology.

The rationale is as follows:

The ontology defines the `vrd:standNextTo` property as an object property that represents a spatial relation where one object is standing next to another object. The ontology also defines the `rdfs:subPropertyOf` property, which is used to declare that one property is a subproperty of another property.

In this case, the triple states that the `vrd:standNextTo` property is a subproperty of itself. This is a valid and meaningful statement in the context of this ontology, as it indicates that the `vrd:standNextTo` property is reflexive - i.e., if an object A stands next to an object B, then object A also stands next to itself.

Reflexivity is a common characteristic of spatial relations, and it is appropriate for the `vrd:standNextTo` property to be modeled as a reflexive property in this ontology. The triple `<vrd:standNextTo> <rdfs:subPropertyOf> <vrd:standNextTo>` captures this reflexive nature of the `vrd:standNextTo` property.

Therefore, the given knowledge graph triple is true in the hypothetical world described by the provided ontology.
Answer: 1
###
Subject: <vrd:drive>
Predicate: <rdfs:domain>
Object: <vrd:NaturalEnvironmentThing>
Rationale: The given knowledge graph triple is true in the hypothetical world described by the provided ontology.

The ontology defines the property 'drive' to have the domain 'Person', which is a subclass of 'Sapiens', 'Mammal', 'Animal', 'LivingEarthBoundThing', 'NaturalEnvironmentEarthBoundThing', and ultimately 'NaturalEnvironmentThing'. 

This means that the 'drive' property can be used to relate instances of the 'NaturalEnvironmentThing' class as the subject, since 'Person' is a subclass of 'NaturalEnvironmentThing'. The knowledge graph triple correctly states that the domain of the 'drive' property is 'NaturalEnvironmentThing', which is a more general class that encompasses the actual domain defined in the ontology.

Therefore, the given knowledge graph triple is true in the context of the provided ontology, as the domain specified in the triple is a superclass of the actual domain defined for the 'drive' property.
Answer: 1
###
Subject: <vrd:sitBehind>
Predicate: <owl:equivalentProperty>
Object: <vrd:sitBehind>
Rationale: The given knowledge graph triple is true in the hypothetical world described by the provided ontology.

The rationale is as follows:

In the ontology, the property `vrd:sitBehind` is defined as a subproperty of `vrd:behind`, which is a transitive property. The ontology also defines the property `vrd:sitBehind` to be equivalent to itself, i.e., `<vrd:sitBehind> <owl:equivalentProperty> <vrd:sitBehind>`.

This means that the property `vrd:sitBehind` is equivalent to itself, as stated in the given knowledge graph triple. The ontology does not state that `vrd:sitBehind` is equivalent to the general `owl:equivalentProperty` relationship, but rather that it is equivalent to itself.

Therefore, the given knowledge graph triple, `<vrd:sitBehind> <owl:equivalentProperty> <vrd:sitBehind>`, is true in the hypothetical world described by the provided ontology.
Answer: 1
###
Subject: <owl:Nothing>
Predicate: <rdfs:subClassOf>
Object: <vrd:ProtectiveDevice>
Rationale: The given knowledge graph triple, <owl:Nothing> <rdfs:subClassOf> <vrd:ProtectiveDevice>, is true in the hypothetical world described by the provided ontology.

The rationale is as follows:

1. In the provided ontology, the class <vrd:ProtectiveDevice> is defined as a subclass of the class <vrd:Device>. This means that all instances of <vrd:ProtectiveDevice> are also instances of <vrd:Device>.

2. The class <owl:Nothing> is a special class in OWL that represents the empty set, i.e., the class that has no instances. By definition, <owl:Nothing> is a subclass of every other class in OWL.

3. Therefore, in the hypothetical world described by the provided ontology, <owl:Nothing> is a subclass of <vrd:ProtectiveDevice>, since <vrd:ProtectiveDevice> is a class defined in the ontology.

In conclusion, the given knowledge graph triple is true in the hypothetical world described by the provided ontology.
Answer: 1
###
Subject: <owl:Nothing>
Predicate: <rdfs:subClassOf>
Object: <vrd:Counter>
Rationale: The given knowledge graph triple stating that <owl:Nothing> is a subclass of <vrd:Counter> is a valid and true statement in the hypothetical world described by the provided ontology.

In the ontology, the class <vrd:Counter> is defined as a subclass of <vrd:FlatSurfaceFurniture>, which is a type of furniture. The class <owl:Nothing> is a special class in OWL that represents the empty set, and it is a subclass of every other class in OWL.

Since <owl:Nothing> is a subclass of every class, it is also a subclass of the class <vrd:Counter>. This is a valid and true statement within the context of the provided ontology.

Therefore, the given knowledge graph triple is true.
Answer: 1
###
Subject: <owl:Nothing>
Predicate: <rdfs:subClassOf>
Object: <vrd:AirMotorisedVehicle>
Rationale: The given knowledge graph triple is true.

The class <owl:Nothing> is a special class in OWL that represents the empty set - the class that has no instances. As per the OWL specification, <owl:Nothing> is a subclass of every other class, including <vrd:AirMotorisedVehicle>. 

This is because the empty set is a subset of any other set, including the set of all air motorized vehicles. Therefore, the statement that <owl:Nothing> is a subclass of <vrd:AirMotorisedVehicle> is logically true.
Answer: 1
###
Subject: <vrd:fly>
Predicate: <rdfs:domain>
Object: <vrd:PullCapableThing>
Rationale: The given knowledge graph triple is true in the hypothetical world described by the provided ontology.

The ontology defines the property 'fly' as a subproperty of 'playWith', with the domain restricted to the class 'Person' and the range restricted to the class 'Kite'. This means that the property 'fly' can be used to describe relationships between instances of the class 'Person' and instances of the class 'Kite'.

The ontology also defines the class 'PullCapableThing' as an equivalent class to the union of the classes 'Horse', 'Person', 'TrainEngine', and 'Truck'. This means that the class 'PullCapableThing' is a superclass of the class 'Person'.

Since the domain of the property 'fly' is a subclass of the class 'PullCapableThing', the given knowledge graph triple is true in the hypothetical world described by the provided ontology.
Answer: 1
###
Subject: <vrd:ride>
Predicate: <owl:equivalentProperty>
Object: <vrd:ride>
Rationale: The given knowledge graph triple is true in the context of the provided ontology.

The ontology defines the property `<vrd:ride>` as follows:

```
<http://www.semanticweb.org/nesy4vrd/ontologies/vrd_world#ride> rdf:type owl:ObjectProperty ;
         rdfs:subPropertyOf vrd:on ;
         rdfs:domain vrd:Person ;
         rdfs:range vrd:RidableThing .
```

This means that the property `<vrd:ride>` is a subproperty of `<vrd:on>`, and it relates a `vrd:Person` to a `vrd:RidableThing`.

The knowledge graph triple states that `<vrd:ride>` is equivalent to `<owl:equivalentProperty>`. This is a valid statement because the `<owl:equivalentProperty>` construct can be used to declare that two properties have the same meaning and can be used interchangeably.

Therefore, the given knowledge graph triple is true in the context of the provided ontology, as it correctly declares the equivalence of the `<vrd:ride>` property with the `<owl:equivalentProperty>` construct.
Answer: 1
###
Subject: <owl:Nothing>
Predicate: <rdfs:subClassOf>
Object: <vrd:NaturalEnvironmentNonEarthBoundThing>
Rationale: The given knowledge graph triple is not valid within the context of the provided ontology. The ontology does not contain a class named <owl:Nothing>, which is a special class in OWL that represents the empty set. Additionally, the ontology does not contain a class named <vrd:NaturalEnvironmentNonEarthBoundThing>. 

Therefore, the statement that <owl:Nothing> is a subclass of <vrd:NaturalEnvironmentNonEarthBoundThing> is not a valid triple within the scope of the provided ontology, and the truth value of this triple should be considered false.
Answer: 1
###
Subject: <vrd:feed>
Predicate: <rdfs:domain>
Object: <vrd:PlayWithCapableThing>
Rationale: The given knowledge graph triple stating that the property 'feed' has the class 'PlayWithCapableThing' as its domain is true in the hypothetical world described by the provided ontology.

The rationale is as follows:

1. The ontology defines the 'feed' property to have the domain of 'Person'. This means that the 'feed' property can be used with instances of the 'Person' class.

2. The ontology also defines the 'PlayWithCapableThing' class as the union of the classes 'Ball', 'Person', 'Phone', and 'Skateboard'. This means that the 'Person' class is a subset of the 'PlayWithCapableThing' class.

3. Since the domain of the 'feed' property is 'Person', and the 'Person' class is a subset of the 'PlayWithCapableThing' class, it follows that the 'feed' property can also be used with instances of the 'PlayWithCapableThing' class.

Therefore, the knowledge graph triple stating that the domain of the 'feed' property is the 'PlayWithCapableThing' class is true in the hypothetical world described by the provided ontology.
Answer: 1
###
Subject: <vrd:Table>
Predicate: <rdfs:subClassOf>
Object: <owl:Thing>
Rationale: The given knowledge graph triple stating that <vrd:Table> is a subclass of <owl:Thing> is true.

In the provided ontology, the class <vrd:Table> is defined as a subclass of <vrd:FlatSurfaceFurniture>, which in turn is a subclass of <vrd:Furniture>. The class <vrd:Furniture> is a subclass of <vrd:EngineeredEnvironmentThing>, which is a subclass of <vrd:VRDWorldThing>.

The class <vrd:VRDWorldThing> is the top-level class in the ontology, representing the domain of objects and relationships in the VRD dataset. Although the ontology does not explicitly state that <vrd:VRDWorldThing> is a subclass of <owl:Thing>, the top-level class in the OWL ontology, this is implied by the semantics of the OWL language.

In OWL, the class <owl:Thing> is the universal class that represents the set of all individuals. Any class defined in an OWL ontology is automatically a subclass of <owl:Thing>, unless explicitly stated otherwise. Therefore, since <vrd:Table> is a subclass of <vrd:VRDWorldThing>, which is the top-level class in the provided ontology, it is also a subclass of <owl:Thing>.

Thus, the given knowledge graph triple stating that <vrd:Table> is a subclass of <owl:Thing> is true.
Answer: 1
###
Subject: <vrd:near>
Predicate: <owl:equivalentProperty>
Object: <vrd:by>
Rationale: The given knowledge graph triple is true in the hypothetical world described by the provided ontology.

The ontology states:

"vrd:near rdf:type owl:ObjectProperty ,
          owl:SymmetricProperty ."
"vrd:by rdf:type owl:ObjectProperty ;
       owl:equivalentProperty vrd:near ;
       rdf:type owl:SymmetricProperty ."

This means that the properties `<vrd:near>` and `<vrd:by>` are declared to be equivalent to each other. The statement `<vrd:near> <owl:equivalentProperty> <vrd:by>` is therefore true in this ontology.

The ontology explicitly states that `<vrd:by>` is equivalent to `<vrd:near>`, which means they represent the same relationship. Therefore, the knowledge graph triple `<vrd:near> <owl:equivalentProperty> <vrd:by>` is a true statement in the context of this ontology.
Answer: 1
###
Subject: <owl:Nothing>
Predicate: <rdfs:subClassOf>
Object: <vrd:Shirt>
Rationale: The given knowledge graph triple stating that <owl:Nothing> is a subclass of <vrd:Shirt> is false. 

The class <owl:Nothing> represents the empty set, which means it has no instances. By definition, the empty set is a subclass of every class, including <vrd:Shirt>. This is because the empty set satisfies the necessary and sufficient conditions to be a subclass of any class - it has no instances, and therefore all instances of <owl:Nothing> are also instances of <vrd:Shirt>.

However, the ontology provided does not contain any statements that would support this triple being true. The ontology defines <vrd:Shirt> as a subclass of <vrd:UpperBodyClothing>, but it does not state that <owl:Nothing> is a subclass of <vrd:Shirt>. Therefore, the given triple cannot be considered true within the context of this ontology.
Answer: 1
###
Subject: <owl:Nothing>
Predicate: <rdfs:subClassOf>
Object: <vrd:Plant>
Rationale: The given knowledge graph triple, <owl:Nothing> <rdfs:subClassOf> <vrd:Plant>, is actually true.

The class <owl:Nothing> is a special class in OWL that represents the empty set - the class that has no instances. By definition, the empty set is a subclass of every other class, including <vrd:Plant>. 

This is because the empty set satisfies the necessary and sufficient conditions to be a subclass of any other class. Specifically, for any class C, if there are no instances of the empty set, then trivially all instances of the empty set are also instances of C. Therefore, the empty set is a subclass of C.

So in the context of the VRD-World ontology, the statement that <owl:Nothing> is a subclass of <vrd:Plant> is logically correct and true. The empty set, being a subclass of everything, is also a subclass of the class <vrd:Plant>, which represents the plant kingdom.

Therefore, the given knowledge graph triple is true.
Answer: 1
###
Subject: <vrd:MotorisedVehicle>
Predicate: <rdfs:subClassOf>
Object: <owl:Thing>
Rationale: The given knowledge graph triple stating that <vrd:MotorisedVehicle> is a subclass of <owl:Thing> is true.

In the provided ontology, the class hierarchy is defined such that all classes are ultimately subclasses of the top-level class <vrd:VRDWorldThing>. This class is not explicitly declared as a subclass of <owl:Thing>, the top-level class in the OWL ontology.

However, the OWL ontology specification states that the class <owl:Thing> is the universal class that contains all individuals. By definition, any class defined in an OWL ontology is a subclass of <owl:Thing>.

Therefore, since <vrd:MotorisedVehicle> is a subclass of <vrd:VRDWorldThing>, which is implicitly a subclass of <owl:Thing>, the given knowledge graph triple stating that <vrd:MotorisedVehicle> is a subclass of <owl:Thing> is true.
Answer: 1
###
Subject: <{s}>
Predicate: <{p}>
Object: <{o}>
Rationale: {{rationale}}
Answer: {{answer}}
"""

    PROMPT = PromptTemplate(input_variables=["s", "p", "o", "graph"], template=PROMPT_TEMPLATE)

    OUTPUT_PARSER = RegexParser(
        regex=r"(?is).*Rationale:\**\s*(.*?)Answer:\**\s*(0|1)",
        output_keys=["rationale", "answer"],
        default_output_key="rationale"
    )
    
    def __init__(self, model="gpt-4-0125-preview", temperature=0.1):
        """
        Initializes an intension-as-classifier.
        
        Parameters:
            model: The name of the model to be used for zero shot CoT classification (default "gpt-4-0125-preview").
            temperature: The temperature parameter for the model (default 0.1).
         """
        super().__init__(self.PROMPT, self.OUTPUT_PARSER, model, temperature)
